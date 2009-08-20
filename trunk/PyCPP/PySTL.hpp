// Simple Conversion Template Library
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

#ifndef SCTL_PYSTL_HPP
#define SCTL_PYSTL_HPP

/// \file   PySTL.hpp
///
/// \brief This header contains converter classes for converting between
/// STL and Python containers.
///
/// \author Damian Eads

#include "converter.hpp"
#include "scalars.hpp"
#include "dispatch.hpp"
#include <vector>
#include <map>
#include <string>

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace PyCPP {


  /// Convert an STL vector to a Python object.
  ///
  template <>
  template <typename FromElem >
  struct Converter<std::vector<FromElem>, PyObject*> {

    static void convert(const std::vector<FromElem> &src, PyObject *&dst) {
      PyListObject *d;
      Converter<std::vector<FromElem>, PyListObject*>::convert(src, d);
      dst = (PyObject*)d;
    }
  };

  template <>
  template <typename FromElem >
  struct Converter<std::vector<FromElem>, PyListObject*> {

    static void convert(const std::vector<FromElem> &src, PyListObject *&dst) {
      /** Let's take the approach of using PyList_Append instead of
	  PyList_SetItem so that any converter function called from here
	  can return flow control to the Python program by a
	  PyDECREF. By preallocating a list of a size greater than zero,
	  is in an inconsistent state until the last SetItem is
	  called.*/

      PyObject *ddst = PyList_New(0);
      /**PyObject *ddst = PyList_New(src.size());**/
      if (ddst == 0) {
	throw std::string("Error when allocating Python list.");
      }
      //int i = 0;
      for (typename std::vector<FromElem>::const_iterator it(src.begin()); it != src.end(); it++) {
	PyObject *elemDst;
	Converter<FromElem, PyObject*>::convert(*it, elemDst);
	PyList_Append(ddst, elemDst);
	//i++;
      }
      dst = (PyListObject*)ddst;
    }
  };


  template <>
  template <typename FromElem>
  struct ContainerLength<std::vector<FromElem> > {
    static int len(const std::vector<FromElem> &src) {
      return src.size();
    }

    static int ndims(const std::vector<FromElem> &src) {
      return 1;
    }

    static int dims(const std::vector<FromElem> &src, const int &i) {
      return src.size();
    }
  };


  template <>
  template <typename KeyElem, typename ValueElem >
  struct Converter<std::map<KeyElem, ValueElem>, PyDictObject*> {

    static void convert(const std::map<KeyElem, ValueElem> &src, PyDictObject *&dst) {
      PyObject *retval = PyDict_New();
      for (typename std::map<KeyElem, ValueElem>::const_iterator it(src.begin()); it != src.end(); it++) {
	PyDict_SetItem(retval,
		       Converter<KeyElem, PyObject*>::convert(it->first),
		       Converter<ValueElem, PyObject*>::convert(it->second));
      }
      dst = (PyDictObject*)retval;
    }

  };

  template <>
  template <typename KeyElem, typename ValueElem>
  struct ContainerLength<std::map<KeyElem, ValueElem> > {
    static int len(const std::map<KeyElem, ValueElem> &src) {
      return src.size();
    }
  };

  template <>
  template <typename ToElem >
  struct Converter<PyListObject*, std::vector<ToElem> > {

    static void convert(PyObject *src, std::vector<ToElem> &dst) {
      convertImpl((PyListObject*)src, dst);
    }

    static void convert(PyListObject *src, std::vector<ToElem> &dst) {
      convertImpl(src, dst);
    }

  private:
    static void convertImpl(PyListObject *src, std::vector<ToElem> &dst) {
      dst.clear();
      const int sz = PyList_Size((PyObject*)src);
      for (int i = 0; i < sz; i++) {
	/** borrowed reference here */
	PyObject *item = PyList_GetItem((PyObject*)src, i);
	dst.push_back(Converter<PyObject*, ToElem>::convert(item));
      }
    }
  };

  template <>
  template <typename E1, typename E2>
  struct ToPython<std::vector<E1>, Structure::List<E2> > {
    PyObject *toPython(const std::vector<E1> &in) {
      PyObject *L = PyList_New(in.size());
      for (int i = 0; i < in.size(); i++) {
	PyList_SetItem(L, i, Py_None);
      }
      for (int i = 0; i < in.size(); i++) {
	PyList_SetItem(L, i, ToPython<E1, E2>(in));
      }
      return L;
    }
  };

  template <>
  template <typename E1, typename E2>
  struct ToPython<std::vector<E1>, Structure::Tuple<E2> > {
    PyObject *toPython(const std::vector<E1> &in) {
      PyObject *Tup = PyTuple_New(in.size());
      for (int i = 0; i < in.size(); i++) {
	PyTuple_SetItem(Tup, i, Py_None);
      }
      for (int i = 0; i < in.size(); i++) {
	PyTuple_SetItem(Tup, i, ToPython<E1, E2>(in));
      }
      return Tup;
    }
  };

  template <>
  template <typename KeyElem, typename ValueElem, typename Comparator, typename PyKey, typename PyValue >
  struct ToPython<std::map<KeyElem, ValueElem, Comparator>, Structure::Dict<PyKey, PyValue> > {
    PyObject *toPython(const std::map<KeyElem, ValueElem, Comparator> &src) {
      PyObject *D = PyDict_New();
      for (typename std::map<KeyElem, ValueElem, Comparator>::const_iterator it(src.begin());
	   it != src.end(); it++) {
	PyDict_SetItem(D,
		       ToPython<KeyElem, PyKey>::toPython(it->first),
		       ToPython<ValueElem, PyValue>::toPython(it->second));
      }
      return D;
    }
  };

}

#endif
