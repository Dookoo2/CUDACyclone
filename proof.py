#!/usr/bin/env python3
import argparse
import subprocess
import secrets
import hashlib
import time
import sys
from typing import List, Tuple, Optional, Set
from ecdsa import SECP256k1, SigningKey
import select

BASE58_ALPH = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def base58_encode(b: bytes) -> str:
    zeros = 0
    for c in b:
        if c == 0:
            zeros += 1
        else:
            break
    num = int.from_bytes(b, "big")
    enc = bytearray()
    while num > 0:
        num, rem = divmod(num, 58)
        enc.append(BASE58_ALPH[rem])
    enc = bytes(reversed(enc))
    return ("1" * zeros) + enc.decode("ascii")


def sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def hash160(b: bytes) -> bytes:
    sha = hashlib.sha256(b).digest()
    h = hashlib.new("ripemd160")
    h.update(sha)
    return h.digest()


def p2pkh_from_pubkey_compressed(pubkey_comp: bytes) -> str:
    h160 = hash160(pubkey_comp)
    payload = b"\x00" + h160
    checksum = sha256d(payload)[:4]
    return base58_encode(payload + checksum)


def compressed_pubkey_from_priv32(priv32: bytes) -> bytes:
    sk = SigningKey.from_string(priv32, curve=SECP256k1)
    vk = sk.get_verifying_key()
    xy = vk.to_string()
    x = xy[:32]
    y = xy[32:]
    prefix = b"\x03" if (int.from_bytes(y, "big") & 1) else b"\x02"
    return prefix + x


def parse_hex_range(range_str: str) -> Tuple[int, int]:
    if ":" not in range_str:
        raise ValueError("Range must be HEX_START:HEX_END")
    s, e = range_str.split(":", 1)
    s = s.strip()
    e = e.strip()
    if s.startswith(("0x", "0X")):
        s = s[2:]
    if e.startswith(("0x", "0X")):
        e = e[2:]
    s = s or "0"
    e = e or "0"
    si = int(s, 16)
    ei = int(e, 16)
    if si > ei:
        raise ValueError("Start > End in range")
    return si, ei


def int_to_priv32_hex(i: int) -> str:
    return i.to_bytes(32, "big").hex()


def generate_unique_privkeys_in_range(start: int, end: int, desired: int) -> List[str]:
    order = SECP256k1.order
    keyset: Set[str] = set()
    attempts = 0
    max_attempts = max(1_000_000, desired * 1000)
    rng_size = end - start + 1
    if rng_size <= 0:
        raise ValueError("Range size <= 0")
    while len(keyset) < desired and attempts < max_attempts:
        if rng_size <= (1 << 62):
            r = secrets.randbelow(rng_size) + start
        else:
            width = end.bit_length()
            r = start + (secrets.randbits(width) % rng_size)
        attempts += 1
        if r == 0 or r >= order:
            continue
        keyset.add(int_to_priv32_hex(r))
    if len(keyset) < desired:
        raise RuntimeError(
            f"Failed to generate {desired} unique private keys (got {len(keyset)} after {attempts} attempts)"
        )
    return list(keyset)


def run_cyclone_and_watch(
    cyclone_path: str,
    range_arg: str,
    address: str,
    grid_arg: Optional[str],
    match_marker: str = "======== FOUND MATCH! =================================",
    timeout: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    args = [cyclone_path, "--range", range_arg, "--address", address]
    if grid_arg:
        args += ["--grid", grid_arg]
    p = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
    )
    found_priv: Optional[str] = None
    start_time = time.time()
    try:
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if match_marker in line:
                for _ in range(20):
                    fd = p.stdout.fileno()
                    rlist, _, _ = select.select([fd], [], [], 0.2)
                    if not rlist:
                        break
                    nxt = p.stdout.readline()
                    if not nxt:
                        break
                    sys.stdout.write(nxt)
                    sys.stdout.flush()
                    if "Private Key" in nxt:
                        parts = nxt.split(":", 1)
                        if len(parts) > 1:
                            found_priv = parts[1].strip()
                            break
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                return True, found_priv
            if "Private Key" in line and found_priv is None:
                parts = line.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    found_priv = parts[1].strip()
            if timeout is not None and (time.time() - start_time) > timeout:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                return False, None
        p.wait()
        if found_priv is not None:
            return True, found_priv
        return False, None
    finally:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cyclone test runner. Generates private keys in range, derives P2PKH, runs CUDACyclone, records results."
    )
    parser.add_argument(
        "--range",
        "-r",
        dest="range_arg",
        required=True,
        help="HEX range START:END (e.g. 80000000:FFFFFFFF)",
    )
    parser.add_argument(
        "--cyclone-path",
        "-c",
        dest="cyclone_path",
        default="./CUDACyclone",
        help="Path to CUDACyclone binary",
    )
    parser.add_argument(
        "--grid",
        dest="grid_arg",
        default="512,256",
        help="Value for --grid passed to CUDACyclone",
    )
    parser.add_argument(
        "--keys",
        dest="keys",
        type=int,
        default=1000,
        help="Number of unique private keys to generate (default: 1000)",
    )
    parser.add_argument(
        "--tests",
        dest="tests",
        type=int,
        default=100,
        help="Number of tests to run (default: 100)",
    )
    parser.add_argument(
        "--timeout",
        dest="timeout",
        type=int,
        default=None,
        help="Optional timeout in seconds for each CUDACyclone run",
    )
    args = parser.parse_args()

    try:
        start_i, end_i = parse_hex_range(args.range_arg)
    except Exception as ex:
        print("Range parse error:", ex, file=sys.stderr)
        sys.exit(1)

    print(f"Generation range for private keys: {hex(start_i)} .. {hex(end_i)}")
    print(f"Generating {args.keys} unique private keys...")
    keys = generate_unique_privkeys_in_range(start_i, end_i, args.keys)
    print(f"Generated {len(keys)} keys.")

    out_fname = "cyclone_tests_results.txt"
    with open(out_fname, "w", encoding="utf-8") as ofs:
        ofs.write(
            f"Cyclone tests results\nRange: {args.range_arg}\nCyclone path: {args.cyclone_path}\nGrid: {args.grid_arg}\nDate: {time.ctime()}\n\n"
        )
        successes = 0
        failures = 0
        for t in range(args.tests):
            priv_hex = keys[t % len(keys)]
            priv_bytes = bytes.fromhex(priv_hex)
            try:
                pub_comp = compressed_pubkey_from_priv32(priv_bytes)
                addr = p2pkh_from_pubkey_compressed(pub_comp)
            except Exception as ex:
                print(f"[{t+1}] Error computing public key/address: {ex}", file=sys.stderr)
                ofs.write(f"{t+1}, {priv_hex}, , ERROR_KEY\n")
                failures += 1
                continue

            print(f"\n=== Test {t+1}/{args.tests} ===\npriv: {priv_hex}\naddress: {addr}\nLaunching Cyclone...")
            ofs.write(f"{t+1}, {priv_hex}, {addr}, START\n")
            ofs.flush()

            found, found_priv = run_cyclone_and_watch(
                args.cyclone_path, args.range_arg, addr, args.grid_arg, timeout=args.timeout
            )
            if found:
                successes += 1
                print(f"*** FOUND on test {t+1} (found_priv from cyclone: {found_priv}) ***")
                ofs.write(f"{t+1}, {priv_hex}, {addr}, FOUND, {found_priv}\n")
            else:
                failures += 1
                print(f"--- No match on test {t+1} ---")
                ofs.write(f"{t+1}, {priv_hex}, {addr}, NO_MATCH\n")
            ofs.flush()
            time.sleep(0.2)

        ofs.write("\nSummary:\n")
        ofs.write(f"Total tests: {args.tests}\nSuccesses: {successes}\nFailures: {failures}\n")
    print(f"\nDone. Results in {out_fname}. Successes={successes} Failures={failures}")


if __name__ == "__main__":
    main()
