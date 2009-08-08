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

#ifndef SCTL_NUMPYSTL_HPP
#define SCTL_NUMPYSTL_HPP

/// \file   PyNumPySTL.hpp
///
/// \brief This header contains converter classes for converting
/// between STL containers and NumPy arrays.
///
/// \author Damian Eads

#include <string>
#include <vector>

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace PyCPP {

  /// A converter class for converting from NumPy arrays to STL
  /// vectors.

  template <>
  template <typename ToElem>
  struct Converter<PyArrayObject*, std::vector<ToElem> > {

    /// Converts a one dimensional NumPy array to an STL vector. The
    /// buffer storing the elements in the NumPy array is not reused
    /// since STL vectors manage their arrays as they see fit.
    ///
    /// @param src  A pointer to the PyArrayObject *.
    /// @param dst  A pointer to the STL vector to store the elements.
    ///             If there are elements already in the STL vector,
    ///             they are not cleared. The elements in the NumPy
    ///             array are appended to the STL vector in order.
    ///
    /// \exception std::string Thrown if the array is not
    ///                        one-dimensional or if an error occurs
    ///                        during NumPy array iteration.

    static void convert(PyArrayObject *src, std::vector<ToElem> &dst) {
      if (PyArray_NDIM(src) != 1) {
	throw std::string("The NumPy array passed must be one-dimensional!");
      }
      if (PyArray_TYPE(src) != NumPyType<ToElem>::type) {
	throw std::string("The NumPy array passed must be of type "
			  + NumPyType<ToElem>::name()
			  + " (type code " + NumPyType<ToElem>::code() + ")!");
      }
      int sz = src->dimensions[0];
      //dst.resize(sz);
      if (PyArray_ISCONTIGUOUS(src)) {
	typedef typename NumPyType<ToElem>::ElemType ElemType;
	ElemType *rdata = (ElemType*)src->data;
	//ToElem to;
	for (int i = 0; i < sz; i++) {
	  dst.push_back(NumPyType<ToElem>::toCPP(rdata[i]));
	}
      }
      else {
	PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	if (iter == 0) {
	  throw std::string("Could not grab iterator!");
	}
	typedef typename NumPyType<ToElem>::ElemType ElemType;
	//ToElem to;
	for (int i = 0; iter->index < iter->size; i++) {
	  dst.push_back(NumPyType<ToElem>::toCPP(*((ElemType*)iter->dataptr)));
	  PyArray_ITER_NEXT(iter);
	}
      }
    }

    /// Converts a one dimensional NumPy array to an STL vector. The
    /// buffer storing the elements in the NumPy array is not reused
    /// since STL vectors manage their arrays as they see fit. This
    /// version of the method is less preferable since the STL vector
    /// must get copied after return.
    ///
    /// @param src  A pointer to the PyArrayObject *.
    ///
    /// \exception std::string Thrown if the array is not
    ///                        one-dimensional or if an error occurs
    ///                        during NumPy array iteration.
    static std::vector<ToElem> convert(PyArrayObject *src) {
      std::vector<ToElem> retval;
      convert(src, retval);
      return retval;
    }
  };



  /// A converter class for converting from NumPy STL vectors to new
  /// one-dimensional NumPy arrays.
  template <>
  template <typename FromElem>
  struct Converter<std::vector<FromElem>, PyArrayObject*> {

    /// Converts an STL vector of type FromElem to a new
    /// one-dimensional NumPy array.
    ///
    /// @param src         The vector to convert.
    /// @param dst         The pointer to the Python array object passed as a
    ///                    reference. This pointer value is changed to
    ///                    point to the new Python array.
    ///
    /// \exception std::string If an error occurs when attempting to allocate a new NumPy array.
    static void convert(const std::vector<FromElem> &src, PyArrayObject *&dst) {
      npy_intp dims[] = {src.size()};
      dst = (PyArrayObject*)PyArray_SimpleNew(1, dims, NumPyType<FromElem>::type);
      if (dst == 0) {
	throw std::string("Error when trying to allocate new NumPy array.");
      }
      // The NumPyElementConverterToPython typedef will enable us to either define a
      //  primitive pointer (like an int*) or an object pointer
      //  (PyObject**).src
      //
      //  Note: need to check whether object arrays are stored as
      //  PyObject* or PyObject**'s
      typedef typename NumPyType<FromElem>::ElemType NumPyElemType;
      NumPyElemType *rdata = (NumPyElemType*)dst->data;
      //(typename NumPyElementConverterToPython<FromElem>::typename ToType*)retval->data;
      for (int i = 0; i < (int)src.size(); i++) {
	NumPyType<FromElem>::toNumPy(src[i], rdata[i]);
      }
    }

    /// Converts an STL vector to a new NumPy array, whose pointer is
    /// returned.
    ///
    /// @param src      The vector to convert.
    /// @return         The pointer to the NumPy array object.
    static PyArrayObject* convert(const std::vector<FromElem> &src) {
      PyArrayObject *retval(0);
      convert(src, retval);x
      return retval;
    }
  };

}

#endif
