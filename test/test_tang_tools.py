#
# This file is part of LiteX-WR-NIC.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#

"""Host logic of the Tang PPS and relink bench tools against simulated CSRs."""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def load(name):
    spec   = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pps    = load("tang_pps")
relink = load("tang_relink")


class Reg:
    def __init__(self, read=None, write=None):
        self._read  = read
        self._write = write

    def read(self):
        return self._read()

    def write(self, value):
        self._write(value)


# PPS ----------------------------------------------------------------------------------------------

class PPSBench:
    """A PPS input whose true edge is ``edge_ps`` after the second; every count read is a new second."""
    def __init__(self, edge_ps, tap_ps=pps.TAP_PS):
        self.edge_ps = edge_ps
        self.tap_ps  = tap_ps
        self.taps    = 0
        self.count   = 0
        self.seconds = 100
        self.writes  = []
        self.regs    = SimpleNamespace(
            pps_timestamper_control = Reg(write=self.clear),
            pps_timestamper_count   = Reg(read=self.tick),
            pps_timestamper_seconds = Reg(read=lambda: self.seconds),
            pps_timestamper_cycles  = Reg(read=lambda: self.position() // pps.CYCLE_PS),
            pps_timestamper_fine    = Reg(read=lambda: 0x100 | self.position() % pps.CYCLE_PS // 1000),
            pps_timestamper_samples = Reg(read=lambda: 0),
            pps_sampler_delay       = Reg(write=self.delay),
        )

    def position(self):
        # Sampled at the first 1 ns sample at or after the delayed edge.
        delayed = self.edge_ps + self.taps*self.tap_ps
        return int(-(-delayed // 1000))*1000

    def tick(self):
        self.count   += 1
        self.seconds += 1
        return self.count

    def clear(self, value):
        assert value == 1
        self.count = 0

    def delay(self, value):
        assert value & (1 << 8), "taps must be loaded"
        self.writes.append(value)
        self.taps = value & 0xff


def test_csr_names_exist_in_gateware():
    # The tools address CSRs by name: keep them in step with the gateware.
    root     = Path(__file__).resolve().parents[1]
    gateware = root / "litex_wr_nic/gateware"
    sources  = {
        "pps_timestamper.py" : ["_control", "_count", "_seconds", "_cycles", "_fine", "_samples", "_delay"],
        "wr_phy.py"          : ["status", "fifo_levels", "link_events", "control"],
    }
    for source, regs in sources.items():
        text = (gateware / source).read_text()
        for reg in regs:
            assert re.search(rf"self\.{reg}\s*=\s*CSR", text), (source, reg)
    target = (root / "tang_mega_138k_pro_wr.py").read_text()
    for name in ("pps_timestamper", "pps_out_timestamper", "pps_sampler", "phy"):
        assert re.search(rf"self\.{name}\s*=", target), name


def test_edge_decodes_offset_inside_second():
    e = pps.edge(seconds=7, cycles=3, fine=0x105, samples=0xe0)
    assert e == dict(seconds=7, cycles=3, index=5, valid=True, samples=0xe0, offset_ps=3*8000 + 5000)
    assert pps.edge(0, 1, 0, oversampling=1)["offset_ps"] == 8000
    assert not pps.edge(0, 0, 0x007)["valid"]


def test_timestamps_and_summary():
    bench = PPSBench(edge_ps=8000 + 2300)
    timestamper = pps.PPSTimestamper(bench)
    timestamper.clear()
    count, edges = 0, []
    for _ in range(4):
        e, count = timestamper.wait(count)
        edges.append(e)
    assert [e["offset_ps"] for e in edges] == [11000]*4
    assert len({e["seconds"] for e in edges}) == 4
    s = pps.summary(edges)
    assert s["spread_ps"] == 0 and s["mean_ps"] == 11000 and s["invalid"] == 0


def test_wait_times_out_without_edge():
    bench = PPSBench(edge_ps=0)
    bench.regs.pps_timestamper_count = Reg(read=lambda: 5)
    with pytest.raises(TimeoutError, match="pps_timestamper"):
        pps.PPSTimestamper(bench).wait(5, timeout=0.05)


@pytest.mark.parametrize("edge_ps", [10_100, 10_400, 10_975])
def test_delay_scan_refines_edge(edge_ps):
    bench = PPSBench(edge_ps=edge_ps)
    timestamper = pps.PPSTimestamper(bench)
    results = pps.scan_delay(timestamper, range(0, 100, 1))
    tap, refined = pps.find_step(results)
    # The refined position is within one tap before the edge.
    assert refined < edge_ps <= refined + pps.TAP_PS
    assert all(w & (1 << 8) for w in bench.writes)


def test_delay_scan_without_step():
    bench = PPSBench(edge_ps=10_000)
    results = pps.scan_delay(pps.PPSTimestamper(bench), [0])
    assert pps.find_step(results) is None


def test_set_delay_rejects_out_of_range():
    with pytest.raises(ValueError):
        pps.set_delay(PPSBench(0), 256)


# Relink -------------------------------------------------------------------------------------------

class PHYBench:
    """A PHY that re-aligns after each reset; ``fifo`` returns successive FIFO levels."""
    def __init__(self, fifo_levels, bitslide=3, alignments=1):
        self.control    = bitslide << 8
        self.up         = True
        self.aligned    = 10
        self.lost       = 4
        self.fifo       = iter(fifo_levels)
        self.alignments = alignments
        self.controls   = []
        self.regs       = SimpleNamespace(
            phy_status      = Reg(read=lambda: 0x1f if self.up else 0x01),
            phy_control     = Reg(read=lambda: self.control, write=self.write_control),
            phy_fifo_levels = Reg(read=lambda: next(self.fifo)),
            phy_link_events = Reg(read=lambda: (self.lost << 16) | self.aligned),
        )

    def write_control(self, value):
        self.controls.append(value)
        self.control = value
        if value & relink.CONTROL_RESET:
            self.up    = False
            self.lost += 1
        elif not self.up:
            self.up       = True
            self.aligned  = (self.aligned + self.alignments) & 0xffff
            self.lost    += self.alignments - 1


def test_decoders():
    assert relink.decode_fifo_levels((14 << 5) | 2) == dict(rx=2, tx=14)
    assert relink.decode_link_events((3 << 16) | 0xffff) == dict(aligned=0xffff, lost=3)
    assert relink.delta16(1, 0xffff) == 2


def test_relink_keeps_bitslide_and_reports_steady_link():
    bench = PHYBench([(14 << 5) | 2]*3)
    phy   = relink.TangPHY(bench)
    results = [relink.relink(phy, timeout=0.5) for _ in range(3)]
    assert all(r["fifo"] == dict(rx=2, tx=14) and r["aligned"] == 1 and r["lost"] == 1 for r in results)
    assert bench.controls == [(3 << 8) | 1, 3 << 8]*3
    assert relink.check(results) == []


def test_relink_flags_changed_fifo_and_extra_alignments():
    bench = PHYBench([(14 << 5) | 2, (13 << 5) | 3], alignments=2)
    phy   = relink.TangPHY(bench)
    results = [relink.relink(phy, timeout=0.5) for _ in range(2)]
    messages = relink.check(results)
    assert any("cycle 1: FIFO levels rx=3 tx=13" in m for m in messages)
    assert sum("comma alignment acquisitions" in m for m in messages) == 2


def test_relink_times_out_when_phy_stays_down():
    bench = PHYBench([0])
    bench.regs.phy_status = Reg(read=lambda: 0x03)
    with pytest.raises(TimeoutError, match="phy_status=0x03"):
        relink.relink(relink.TangPHY(bench), timeout=0.05)
