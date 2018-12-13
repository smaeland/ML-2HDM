/* SWIG module */

%module Analysis

%include <std_string.i>
%include "std_vector.i"
%template(VectorDouble) std::vector<double>;
%{
#define SWIG_FILE_WITH_INIT
#include "analysis.h"
%}
%include "analysis.h"
