Every pinned-local timing on record was taken on one M4 P core, and the reason
the top-k census's extra coverage was dismissed is that read-mostly slot loads
are independent and hidden by the out-of-order core. That is a wide-core
argument, so it was tested by pricing l1 on other cores.

The probe drops arm64 alone to the reduced class set, leaving x86-64 on the full
set as an untouched control; the x64 jobs duly reported byte-identical
executables on both sides, so every arm64 delta comes from that one change. The
candidate is the build WITHOUT l1, so a negative delta is what l1 is worth on
that core. Spec suites stayed green at 174/174 interp and 257/257 JIT.

Gate-confirmed rows, arm64-darwin (macos-14) / interp:

    metric            baseline    no-l1     delta      pair volatility
    lz4-decompress      379.24   334.68   -11.75%             2.43%
    lz4-compress        220.09   206.63    -6.11%             1.46%
    lua-sunfish        1,040.2   992.33    -4.60%             2.22%
    sha256               23.25    22.22    -4.46%             2.13%
    coremark           5,274.0  5,058.4    -4.09%             1.40%

Gate-confirmed rows, arm64-linux (ubuntu-24.04-arm) / interp:

    lz4-decompress      190.72   175.87    -7.79%             4.50%
    coremark           3,138.8  2,994.7    -4.59%             0.90%
    bzip2                1.891    1.833    -3.07%             0.67%
    lua-sunfish         696.78   705.95    +1.32%  IMPROVEMENT 0.80%
    lua-json           1,495.9  1,506.2    +0.69%  IMPROVEMENT 0.39%

The width hypothesis does not survive. CoreMark prices l1 at 4.09% on the M1
runner and 4.59% on the Neoverse runner against the 4.2% already measured on the
M4 P core -- the same number on three cores of different width, so nothing about
narrowness changes what a pin is worth there.

What does not survive alongside it is the scope of the l1 verdict. CoreMark is
the weakest metric in the confirmed set: on the same M1, l1 is worth 4.5% to
11.8% on five of fifteen metrics, and lz4-decompress alone is 2.9x CoreMark.
Every prior l1 figure -- the +4.2% over l0, the +1..+3% net bound, the "unproven
against its 2x handler size" -- was measured on CoreMark alone and therefore
priced the change at its corpus minimum.

The two results together contradict the payoff law. That law says a pin converts
only where it breaks a binding loop-carried chain, and the corpus chain census
puts lz4's hot loops at 0.97 effective independent chains with only 8.3% of them
carrying two or more -- so a second pin should have almost nothing to break
there. It is nevertheless worth 11.75% on lz4-decompress at 2.43% pair
volatility. Whatever pays that is not chain criticality.

Two things bound the reading. The probe removes l1 and with it the handler
multiplication, taking the arm64 engine from 339,064 to 177,924 bytes -- under a
192 KB L1I for the first time -- so each delta is a net of the residency loss
against that size refund, and the residency component alone is larger than the
figure shown. And the two confirmed improvements on arm64-linux are the same
effect with the sign reversed: on lua-sunfish and lua-json the smaller engine
beats the lost residency outright.
