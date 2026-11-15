#!/usr/bin/env python3

#
# This file is part of LiteX-WR-NIC.
#
# Copyright (c) 2026 Enjoy-Digital <enjoy-digital.fr>
# SPDX-License-Identifier: BSD-2-Clause
#

"""Re-acquire the Tang Mega 138K Pro SerDes link and report its receive-path state.

Each cycle pulses ``phy_control.reset`` (keeping the ``bitslide`` offset), which resets the
SerDes and PCS and forces CDR lock and comma alignment again, waits for ``phy_status`` to report
the PHY ready, then reports the SerDes FIFO occupancies (``phy_fifo_levels``) and the comma
alignment acquisitions/losses (``phy_link_events``). A FIFO occupancy that differs between
cycles, or more than one alignment per cycle, flags a re-acquisition that may move the receive
latency.

The Gowin word aligner does not report its bit slip, so the latency itself is read from WR:
after each cycle the tool waits ``--settle`` seconds for the WR handshake to complete; read the
round-trip delay (``crtt``) with ``stat`` in the WR console meanwhile (with the peer as master).

Start litex_server on the FPGA UART first:

    litex_server --uart --uart-port /dev/serial/by-id/FPGA_UART --uart-baudrate 115200
"""

import argparse
import time
from pathlib import Path

# Constants ----------------------------------------------------------------------------------------

STATUS_PLL_LOCK = 1 << 0
STATUS_CDR_LOCK = 1 << 1
STATUS_ALIGNED  = 1 << 2
STATUS_RX_VALID = 1 << 3
STATUS_READY    = 1 << 4
STATUS_UP       = STATUS_PLL_LOCK | STATUS_CDR_LOCK | STATUS_ALIGNED | STATUS_READY

CONTROL_RESET    = 1 << 0
CONTROL_BITSLIDE = 0xf << 8

# Decoders -----------------------------------------------------------------------------------------

def decode_fifo_levels(value):
    return dict(rx=value & 0x1f, tx=(value >> 5) & 0x1f)

def decode_link_events(value):
    return dict(aligned=value & 0xffff, lost=(value >> 16) & 0xffff)

def delta16(new, old):
    return (new - old) & 0xffff

# PHY ----------------------------------------------------------------------------------------------

class TangPHY:
    """Host side of ``GW5WRPHY`` (``phy_*`` CSRs)."""
    def __init__(self, bus):
        self.bus = bus

    def status(self):
        return self.bus.regs.phy_status.read()

    def fifo_levels(self):
        return decode_fifo_levels(self.bus.regs.phy_fifo_levels.read())

    def link_events(self):
        return decode_link_events(self.bus.regs.phy_link_events.read())

    def reset(self, hold=0.01):
        control = self.bus.regs.phy_control.read() & CONTROL_BITSLIDE
        self.bus.regs.phy_control.write(control | CONTROL_RESET)
        time.sleep(hold)
        self.bus.regs.phy_control.write(control)

    def wait_up(self, timeout=5.0):
        start    = time.monotonic()
        deadline = start + timeout
        while True:
            status = self.status()
            if status & STATUS_UP == STATUS_UP:
                return time.monotonic() - start
            if time.monotonic() > deadline:
                raise TimeoutError(f"PHY not ready after {timeout} s (phy_status=0x{status:02x})")
            time.sleep(0.01)

def relink(phy, timeout=5.0):
    """One reset/re-acquisition; returns the PHY state after it."""
    before = phy.link_events()
    phy.reset()
    elapsed = phy.wait_up(timeout)
    after   = phy.link_events()
    return dict(
        up_s    = elapsed,
        fifo    = phy.fifo_levels(),
        aligned = delta16(after["aligned"], before["aligned"]),
        lost    = delta16(after["lost"],    before["lost"]),
    )

def check(results):
    """Messages for cycles whose receive path differs from the first one."""
    messages = []
    if not results:
        return messages
    reference = results[0]["fifo"]
    for i, r in enumerate(results):
        if r["fifo"] != reference:
            messages.append(f"cycle {i}: FIFO levels rx={r['fifo']['rx']} tx={r['fifo']['tx']} "
                            f"differ from rx={reference['rx']} tx={reference['tx']}")
        if r["aligned"] != 1:
            messages.append(f"cycle {i}: {r['aligned']} comma alignment acquisitions (expected 1)")
    return messages

# Main ---------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build",   default="build/tang_mega_138k_pro_wr", help="Build output directory (with csr.csv).")
    parser.add_argument("--host",    default="localhost", help="litex_server host.")
    parser.add_argument("--port",    default=1234, type=int, help="litex_server port.")
    parser.add_argument("--cycles",  default=3, type=int, help="Reset/re-acquisition cycles.")
    parser.add_argument("--settle",  default=30.0, type=float, help="Seconds after each cycle for the WR handshake.")
    parser.add_argument("--timeout", default=5.0, type=float, help="Seconds to wait for the PHY to be ready.")
    args = parser.parse_args()

    if args.cycles < 1:
        parser.error("--cycles must be positive.")

    from litex import RemoteClient

    bus = RemoteClient(host=args.host, port=args.port, csr_csv=str(Path(args.build) / "csr.csv"))
    bus.open()
    try:
        phy     = TangPHY(bus)
        control = bus.regs.phy_control.read()
        events  = phy.link_events()
        fifo    = phy.fifo_levels()
        print(f"Initial: phy_status=0x{phy.status():02x} bitslide={(control >> 8) & 0xf} "
              f"fifo rx={fifo['rx']} tx={fifo['tx']} "
              f"aligned={events['aligned']} lost={events['lost']}")
        results = []
        for cycle in range(args.cycles):
            r = relink(phy, args.timeout)
            results.append(r)
            print(f"Cycle {cycle}: ready after {r['up_s']*1000:.0f} ms, "
                  f"fifo rx={r['fifo']['rx']} tx={r['fifo']['tx']}, "
                  f"+{r['aligned']} aligned, +{r['lost']} lost; "
                  f"read crtt with 'stat' in the WR console.", flush=True)
            time.sleep(args.settle)
        messages = check(results)
        for message in messages:
            print(f"Warning: {message}")
        if not messages:
            print(f"{args.cycles} cycles: FIFO levels and alignment unchanged.")
    finally:
        bus.close()

if __name__ == "__main__":
    main()
