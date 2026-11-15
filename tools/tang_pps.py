#!/usr/bin/env python3

#
# This file is part of LiteX-WR-NIC.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#

"""Read the Tang Mega 138K Pro PPS timestampers through litex_server.

Each rising edge of the PPS input (PMOD0 pin 1) latches the WR seconds/cycles of the reference
cycle in which it was sampled and the index of its first high sample (eight per 8 ns cycle):
the offset inside the second is ``cycles x 8 ns + index x 1 ns``. ``--output`` reads the
timestamper on the generated PPS output instead (one sample per cycle, index always 0).

``--scan-delay`` steps the IODELAY in front of the input sampler (~12.5 ps per tap) and records
the tap at which the timestamp moves to the next sample: the edge sat that many taps before
the sample it was first reported at, which refines the position below 1 ns. The scan takes
two seconds per tap setting; use ``--tap-step`` to trade resolution for time.

Start litex_server on the FPGA UART first:

    litex_server --uart --uart-port /dev/serial/by-id/FPGA_UART --uart-baudrate 115200
"""

import argparse
import statistics
import time
from pathlib import Path

# Constants ----------------------------------------------------------------------------------------

CYCLE_PS       = 8000  # 125 MHz WR reference cycle.
SAMPLES        = 8     # IDES8 samples per reference cycle on the PPS input.
TAP_PS         = 12.5  # Nominal GW5A IODELAY step.
TAPS           = 256
PPS_TIMEOUT    = 2.5   # One edge per second is expected (s).

# PPS Timestamper ----------------------------------------------------------------------------------

class PPSTimestamper:
    """Host side of a ``PPSTimestamper`` (``<name>_*`` CSRs)."""
    def __init__(self, bus, name="pps_timestamper", samples=SAMPLES):
        self.bus     = bus
        self.name    = name
        self.samples = samples

    def _reg(self, reg):
        return getattr(self.bus.regs, f"{self.name}_{reg}")

    def clear(self):
        self._reg("control").write(1)

    def count(self):
        return self._reg("count").read()

    def latest(self):
        seconds = self._reg("seconds").read()
        cycles  = self._reg("cycles").read()
        fine    = self._reg("fine").read()
        return edge(seconds, cycles, fine, self._reg("samples").read(), self.samples)

    def wait(self, count, timeout=PPS_TIMEOUT):
        """Wait for an edge after ``count`` edges and return it with the new count."""
        deadline = time.monotonic() + timeout
        while True:
            new = self.count()
            if new != count:
                return self.latest(), new
            if time.monotonic() > deadline:
                raise TimeoutError(f"No PPS edge on {self.name} within {timeout} s")
            time.sleep(0.02)

def edge(seconds, cycles, fine, samples=0, oversampling=SAMPLES):
    """Decode a latched timestamp; ``offset_ps`` is the position inside the WR second."""
    index = fine & 0xff
    return dict(
        seconds   = seconds,
        cycles    = cycles,
        index     = index,
        valid     = bool((fine >> 8) & 1),
        samples   = samples,
        offset_ps = cycles*CYCLE_PS + index*(CYCLE_PS//oversampling),
    )

def summary(edges):
    """Mean offset and peak-to-peak spread of consecutive timestamps, in ps."""
    offsets = [e["offset_ps"] for e in edges]
    return dict(
        count    = len(offsets),
        mean_ps  = statistics.fmean(offsets),
        min_ps   = min(offsets),
        max_ps   = max(offsets),
        spread_ps = max(offsets) - min(offsets),
        stdev_ps = statistics.pstdev(offsets),
        invalid  = sum(not e["valid"] for e in edges),
    )

# Delay Scan ---------------------------------------------------------------------------------------

def set_delay(bus, taps):
    if not 0 <= taps < TAPS:
        raise ValueError(f"IODELAY taps must be in 0..{TAPS - 1}")
    bus.regs.pps_sampler_delay.write(taps | (1 << 8)) # taps + load pulse.

def scan_delay(timestamper, taps, timeout=PPS_TIMEOUT):
    """Timestamp one edge per IODELAY setting; returns ``[(taps, edge), ...]``."""
    results = []
    count   = timestamper.count()
    for tap in taps:
        set_delay(timestamper.bus, tap)
        # Discard the edge that may have been sampled while the delay changed.
        _,      count = timestamper.wait(count, timeout)
        sample, count = timestamper.wait(count, timeout)
        results.append((tap, sample))
    return results

def find_step(results):
    """First tap whose timestamp moves to a later sample than at the first tap.

    An edge is reported at the first sample that sees it high, so at the first tap it lies at
    most one sample period before the reported position. The delay that pushes it past that
    sample is the distance from the edge to it: the edge sits at the first tap's position minus
    ``tap x TAP_PS`` (to one tap). Returns ``(tap, refined_offset_ps)``, ``None`` without a step.
    """
    first_tap, first = results[0]
    for tap, sample in results[1:]:
        if sample["offset_ps"] > first["offset_ps"]:
            return tap, first["offset_ps"] - (tap - first_tap)*TAP_PS
    return None

# Main ---------------------------------------------------------------------------------------------

def format_edge(e):
    return (f"sec={e['seconds']:>10d} cycles={e['cycles']:>9d} index={e['index']} "
            f"valid={int(e['valid'])} samples=0x{e['samples']:02x} offset={e['offset_ps']/1000:.3f} ns")

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build",      default="build/tang_mega_138k_pro_wr", help="Build output directory (with csr.csv).")
    parser.add_argument("--host",       default="localhost", help="litex_server host.")
    parser.add_argument("--port",       default=1234, type=int, help="litex_server port.")
    parser.add_argument("--samples",    default=10, type=int, help="Edges to timestamp.")
    parser.add_argument("--output",     action="store_true", help="Read the generated PPS output timestamper.")
    parser.add_argument("--scan-delay", action="store_true", help="Scan the PPS input IODELAY taps.")
    parser.add_argument("--tap-step",   default=8, type=int, help="Tap step of the delay scan.")
    parser.add_argument("--timeout",    default=PPS_TIMEOUT, type=float, help="Seconds to wait for each edge.")
    args = parser.parse_args()

    if args.scan_delay and args.output:
        parser.error("--scan-delay applies to the PPS input only.")
    if args.samples < 1 or args.tap_step < 1:
        parser.error("--samples and --tap-step must be positive.")

    from litex import RemoteClient

    bus = RemoteClient(host=args.host, port=args.port, csr_csv=str(Path(args.build) / "csr.csv"))
    bus.open()
    try:
        if args.output:
            timestamper = PPSTimestamper(bus, "pps_out_timestamper", samples=1)
        else:
            timestamper = PPSTimestamper(bus, "pps_timestamper")
        print(f"{timestamper.name}: raw samples=0x{timestamper.latest()['samples']:02x}")

        if args.scan_delay:
            try:
                results = scan_delay(timestamper, range(0, TAPS, args.tap_step), args.timeout)
            finally:
                set_delay(bus, 0)
            for tap, sample in results:
                print(f"taps={tap:>3d} ({tap*TAP_PS:7.1f} ps) {format_edge(sample)}")
            step = find_step(results)
            if step is None:
                print(f"No sample step within {results[-1][0]*TAP_PS:.0f} ps of delay.")
            else:
                print(f"Step at {step[0]} taps: edge at {step[1]/1000:.4f} ns in the second "
                      f"(nominal {TAP_PS} ps/tap).")
            return

        timestamper.clear()
        count = 0
        edges = []
        for _ in range(args.samples):
            sample, count = timestamper.wait(count, args.timeout)
            edges.append(sample)
            print(format_edge(sample))
        s = summary(edges)
        print(f"{s['count']} edges: mean={s['mean_ps']/1000:.3f} ns, "
              f"spread={s['spread_ps']/1000:.3f} ns, stdev={s['stdev_ps']/1000:.3f} ns, "
              f"invalid WR time={s['invalid']}.")
    finally:
        bus.close()

if __name__ == "__main__":
    main()
