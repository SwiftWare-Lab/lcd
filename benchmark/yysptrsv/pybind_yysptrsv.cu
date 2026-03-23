
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "common.h"
#include "mmio.h"
#include "read_mtx.h"
#include "tranpose.h"
#include "YYSpTRSV.h"


namespace py = pybind11;


PYBIND11_MODULE(yysptrsv_wrapper, m) {
    m.def("YYSpTRSV", &YYSpTRSV);
}