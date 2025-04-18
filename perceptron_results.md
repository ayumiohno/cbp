
# tage
### FP
```
-------------------------------------DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation i.e. Counts Not Reset When Warmup Ends)-------------------------------------
       Instr       Cycles      IPC      NumBr     MispBr BrPerCyc MispBrPerCyc        MR     MPKI      CycWP   CycWPAvg   CycWPPKI
      997741       193571   5.1544     111265       1140   0.5748       0.0059   1.0246%   1.1426      63362    55.5807    63.5055
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
### INT
```
-------------------------------------DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation i.e. Counts Not Reset When Warmup Ends)-------------------------------------
       Instr       Cycles      IPC      NumBr     MispBr BrPerCyc MispBrPerCyc        MR     MPKI      CycWP   CycWPAvg   CycWPPKI
      997301       338118   2.9496     128874        265   0.3812       0.0008   0.2056%   0.2657      33919   127.9962    34.0108
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

# Perceptron
based on https://github.com/taraeicher/PerceptronBranchPredictor/blob/master/perceptron.cc and ChatGPT.
## params
- perceptron: 1024
- history: 32
- bias/weight: 16-bit
- total: 64KB

### FP
```
-------------------------------------DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation i.e. Counts Not Reset When Warmup Ends)-------------------------------------
       Instr       Cycles      IPC      NumBr     MispBr BrPerCyc MispBrPerCyc        MR     MPKI      CycWP   CycWPAvg   CycWPPKI
      997741       205300   4.8599     111265       1737   0.5420       0.0085   1.5611%   1.7409      77446    44.5861    77.6213
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
### INT
```
-------------------------------------DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation i.e. Counts Not Reset When Warmup Ends)-------------------------------------
       Instr       Cycles      IPC      NumBr     MispBr BrPerCyc MispBrPerCyc        MR     MPKI      CycWP   CycWPAvg   CycWPPKI
      997301       340056   2.9328     128874        382   0.3790       0.0011   0.2964%   0.3830      33559    87.8508    33.6498
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## when combined with tage
- when the abs(dot) > 2 * thres (confident), use perceptron predictor

Outperformed over tage-only implementaion for FP, but I think this is coincidence.

### FP
```
-------------------------------------DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation i.e. Counts Not Reset When Warmup Ends)-------------------------------------
       Instr       Cycles      IPC      NumBr     MispBr BrPerCyc MispBrPerCyc        MR     MPKI      CycWP   CycWPAvg   CycWPPKI
      997741       194136   5.1394     111265       1134   0.5731       0.0058   1.0192%   1.1366      63933    56.3783    64.0778
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
### INT
```
-------------------------------------DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation i.e. Counts Not Reset When Warmup Ends)-------------------------------------
       Instr       Cycles      IPC      NumBr     MispBr BrPerCyc MispBrPerCyc        MR     MPKI      CycWP   CycWPAvg   CycWPPKI
      997301       338118   2.9496     128874        265   0.3812       0.0008   0.2056%   0.2657      33919   127.9962    34.0108
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
# Piecewise Perceptron
## params
- perceptron: 512
- history: 32
- bias/weight: 16-bit
- mask: 2**4
- total: 512KB

The effectiveness of the piecewise predictor remains limited, even with increased memory usage.
### FP
```
-------------------------------------DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation i.e. Counts Not Reset When Warmup Ends)-------------------------------------
       Instr       Cycles      IPC      NumBr     MispBr BrPerCyc MispBrPerCyc        MR     MPKI      CycWP   CycWPAvg   CycWPPKI
      997741       203923   4.8927     111265       1671   0.5456       0.0082   1.5018%   1.6748      75879    45.4093    76.0508
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```
### INT
```
-------------------------------------DIRECT CONDITIONAL BRANCH PREDICTION MEASUREMENTS (Full Simulation i.e. Counts Not Reset When Warmup Ends)-------------------------------------
       Instr       Cycles      IPC      NumBr     MispBr BrPerCyc MispBrPerCyc        MR     MPKI      CycWP   CycWPAvg   CycWPPKI
      997301       338106   2.9497     128874        324   0.3812       0.0010   0.2514%   0.3249      32959   101.7253    33.0482
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```