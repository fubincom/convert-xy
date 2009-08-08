// Python/C++ Interface Template Library
//
// Copyright (C) 2009 Damian Eads
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SCTL_SCALARS_H
#define SCTL_SCALARS_H

/// \file   scalars.hpp
///
/// \brief This header contains converter classes for converting
/// between Python and C scalars, generically.
///
/// \author Damian Eads

#include <string>

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace PyCPP {

  template <>
  struct Converter<int, PyObject*> {
    static void convert(const int &src, PyObject *&dst) {
      dst = PyInt_FromLong(src);
    }

    static PyObject *convert(const int &src) {
      return PyInt_FromLong(src);
    }
  };

  template <>
  struct Converter<PyObject*, int> {
    static void convert(PyObject *src, int &dst) {
      dst = PyInt_AsLong(src);
    }

    static int convert(PyObject *src) {
      return PyInt_AsLong(src);
    }
  };

  template <>
  struct Converter<long, PyObject*> {
    static void convert(const long &src, PyObject *&dst) {
      dst = PyLong_FromLong(src);
    }

    static PyObject *convert(const long &src) {
      return PyLong_FromLong(src);
    }
  };

  template <>
  struct Converter<PyObject*, long> {
    static void convert(PyObject* src, int& dst) {
      dst = PyLong_AsLong(src);
    }

    static int convert(PyObject* src) {
      return PyLong_AsLong(src);
    }
  };

  template <>
  struct Converter<float, PyObject*> {
    static void convert(const float &src, PyObject *&dst) {
      dst = PyFloat_FromDouble(src);
      if (dst == 0) {
	throw std::string("Error when allocating Python double!");
      }
    }

    static PyObject *convert(const float &src) {
      PyObject *dst = PyFloat_FromDouble(src);
      if (dst == 0) {
	throw std::string("Error when allocating Python double!");
      }
      return dst;
    }
  };

  template <>
  struct Converter<PyObject*, float> {
    static void convert(PyObject* src, float &dst) {
      dst = (float)PyFloat_AsDouble(src);
    }

    static float convert(PyObject* src) {
      return (float)PyFloat_AsDouble(src);
    }
  };

  template <>
  struct Converter<double, PyObject*> {
    static void convert(const double &src, PyObject *&dst) {
      dst = PyFloat_FromDouble(src);
      if (dst == 0) {
	throw std::string("Error when allocating Python double!");
      }
    }

    static PyObject *convert(const double &d) {
      PyObject *dst = PyFloat_FromDouble(d);
      if (dst == 0) {
	throw std::string("Error when allocating Python double!");
      }
      return dst;
    }
  };

  template <>
  struct Converter<PyObject*, double> {
    static void convert(PyObject* src, double &dst) {
      dst = PyFloat_AsDouble(src);
    }

    static double convert(PyObject* src) {
      return PyFloat_AsDouble(src);
    }
  };

  template <>
  struct Converter<std::string, PyObject*> {
    static void convert(const std::string &src, PyObject *&dst) {
      dst = PyString_FromString(src.c_str());
      if (dst == 0) {
	throw std::string("Error when allocating Python string!");
      }
    }

    static PyObject *convert(const std::string &src) {
      return PyString_FromString(src.c_str());
    }
  };

  template <>
  struct Converter<std::string, PyStringObject*> {
    static void convert(const std::string &src, PyStringObject *&dst) {
      dst = (PyStringObject*)PyString_FromString(src.c_str());
      if (dst == 0) {
	throw std::string("Error when allocating Python string!");
      }
    }

    static PyStringObject *convert(const std::string &src) {
      return (PyStringObject*)PyString_FromString(src.c_str());
    }
  };

  template <>
  struct Converter<PyStringObject*, std::string> {
    static void convert(PyStringObject *src, std::string &dst) {
      dst = std::string(PyString_AsString((PyObject*)src));
    }

    static std::string convert(PyStringObject *src) {
      return std::string(PyString_AsString((PyObject*)src));
    }
  };

  /** convert any python object to a string representation.
      This might be dangerous because the control flow
      can go back to the Python program. If the list*/
  template <>
  struct Converter<PyObject*, std::string> {
    static void convert(PyObject *src, std::string &dst) {
      PyStringObject *str = (PyStringObject*)PyObject_Str(src);
      dst = Converter<PyStringObject*, std::string>::convert(str);
      Py_XDECREF(str);
    }

    static std::string convert(PyObject *src) {
      PyStringObject *str = (PyStringObject*)PyObject_Str(src);
      std::string dst;
      Converter<PyStringObject*, std::string>::convert(str, dst);
      Py_XDECREF(str);
      return dst;
    }
  };

  struct End{};

  template<class C, class D> struct TypeList {
    typedef C type;
    typedef D next;
  };

  typedef TypeList<float,
		   TypeList<double, End> > FloatTypes;

}

#endif
