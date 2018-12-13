# Pythia analysis code

## Compile and test:
The analysis code is written in C++, but interfaced to Python using SWIG. To
compile the library, make sure to have a working installation of Pythia
(recommended version is 8.219 or newer), and that the correct path is set in
`env.sh`. If editing the C++ files one also needs to regenerate the SWIG
wrapper files; in this case, add the path to the SWIG `bin` directory in
`compile.sh`. Otherwise, SWIG is not needed. In the `cpp` directory, compile
with
```
bash compile.sh
```

To test the newly created library, do:
```
$ python
>>> import Analysis
>>> a = Analysis.Analysis('test')
```

If running on MacOS and it fails with `unsafe use of relative rpath libpythia8.dylib`, try
```
install_name_tool -change libpythia8.dylib <path to pythia8219>/lib/libpythia8.dylib _Analysis.so
```
and run the test again.
