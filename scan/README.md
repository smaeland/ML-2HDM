
## Prerequisites:
Requires working installations of 2HDMC (1.7.0), HiggsBounds (4.3.1), and
HiggsSignals (1.4.0). Version numbers are recommendations; Newer versions
are expected to work as long as the paths in the Makefile are adjusted
correspondingly.

## Scan:
Scan over mass, tanb and m_12^2 to get cross sections times branching ratio for
A and H, to reproduce figure 7 in appendix. 

Compile:
```
make
```

Start scan, then plot results:
```
python ratio_scanner.py --scan --ncpu [NUM CPUs]
python ratio_scanner.py --plot
```

