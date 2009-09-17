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

#ifndef SCTL_PYCVD_HPP
#define SCTL_PYCVD_HPP

/// \file   PyCVD.hpp
///
/// \brief This header contains converter classes for converting between
/// LIBCVD data structures and NumPy arrays.
///
/// \author Damian Eads

#include <Python.h>
#include "NumPyTypes.hpp"
#include "converter.hpp"
#include "scalars.hpp"
#include <cvd/image.h>
#include <vector>

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace PyCPP {

  template <>
  struct Converter<std::vector<CVD::ImageRef>, PyArrayObject*> {
    static void convert(const std::vector <CVD::ImageRef> &src,
			PyArrayObject *&dst) {
      npy_intp dims[] = {src.size(), 2};
      dst = (PyArrayObject*)PyArray_SimpleNew(2, dims, NumPyType<int>::type);
      if (dst == 0) {
	throw std::string("Error when allocating numpy array of locations.");
      }
      int *rdata = (int*)dst->data;
      for (int i = 0; i < src.size(); i++) {
	rdata[i*2] = src[i].y;
	rdata[i*2+1] = src[i].x;
      }
    }

    static PyArrayObject* convert(const std::vector<CVD::ImageRef> &src) {
      PyArrayObject *retval;
      convert(src, retval);
      return retval;
    }
  };

  template <>
  struct Converter<PyArrayObject*, std::vector<CVD::ImageRef> > {
    static void convert(PyArrayObject *src, std::vector <CVD::ImageRef> &dst) {
      if (PyArray_NDIM(src) != 2) {
	throw std::string("The NumPy array passed must be two-dimensional to convert to a vector<ImageRef>!");
      }
      if (PyArray_TYPE(src) != NumPyType<int>::type) {
	throw std::string("The NumPy array passed must be of type "
			  + NumPyType<int>::name()
			  + " (type code " + NumPyType<int>::code() + ") to convert to a vector<ImageRef>!");
      }
      int nrows = src->dimensions[0];
      int ncols = src->dimensions[1];
      if (ncols != 2) {
	throw std::string("There must be exactly two columns in a coordinate array to convert to a vector<ImageRef>");
      }
      const int *rdata = (int*)src->data;
      for (int i = 0; i < nrows; i++, rdata += 2) {
	dst.push_back(CVD::ImageRef(*(rdata+1), *(rdata)));
      }
    }

    static std::vector<CVD::ImageRef> convert(PyArrayObject *src) {
      std::vector<CVD::ImageRef> retval;
      convert(src, retval);
      return retval;
    }
  };

  template <>
  struct Converter<CVD::ImageRef, PyListObject*> {
    static void convert(const CVD::ImageRef &src, PyListObject *&dst) {
      PyObject *retval = PyList_New(2);
      PyList_SetItem(retval, 0, Converter<int, PyObject*>::convert(src.y));
      PyList_SetItem(retval, 1, Converter<int, PyObject*>::convert(src.x));
      dst = (PyListObject*)retval;
    }

    static PyListObject* convert(const CVD::ImageRef &src) {
      PyListObject *lst;
      convert(src, lst);
      return lst;
    }
  };

  template <>
  struct Converter<CVD::ImageRef, PyTupleObject*> {
    static void convert(const CVD::ImageRef &src, PyTupleObject *&dst) {
      PyObject *retval = PyList_New(2);
      PyTuple_SetItem(retval, 0, Converter<int, PyObject*>::convert(src.y));
      PyTuple_SetItem(retval, 1, Converter<int, PyObject*>::convert(src.x));
      dst = (PyTupleObject*)retval;
    }

    static PyTupleObject* convert(const CVD::ImageRef &src) {
      PyTupleObject *tup;
      convert(src, tup);
      return tup;
    }
  };

  template <>
  struct Converter<CVD::ImageRef, PyObject*> {
    static void convert(const CVD::ImageRef &src, PyObject *&dst) {
      PyTupleObject *tup;
      Converter<CVD::ImageRef, PyTupleObject*>::convert(src, tup);
      dst = (PyObject*)tup;
    }

    static PyObject* convert(const CVD::ImageRef &src) {
      PyTupleObject *tup;
      Converter<CVD::ImageRef, PyTupleObject*>::convert(src, tup);
      return (PyObject*)tup;
    }
  };

  template <>
  template <typename ToElem>
  struct Converter<PyArrayObject*, CVD::Image<ToElem> > {

    static void convert(PyArrayObject *src, CVD::Image<ToElem> &dst) {
      if (PyArray_NDIM(src) != 2) {
	throw std::string("The NumPy array passed must be two-dimensional!");
      }
      if (PyArray_TYPE(src) != NumPyType<ToElem>::type) {
	throw std::string("The NumPy array passed must be of type "
			  + NumPyType<ToElem>::name()
			  + " (type code " + NumPyType<ToElem>::code() + ")!");
      }
      int nrows = src->dimensions[0];
      int ncols = src->dimensions[1];
      if (dst.size().y != nrows && dst.size().x != ncols) {
	dst.resize(CVD::ImageRef(ncols, nrows));
      }
      if (PyArray_ISCONTIGUOUS(src)) {
	typedef typename NumPyType<ToElem>::ElemType ElemType;
	ElemType *rdata = (ElemType*)src->data;
	for (int i = 0, k = 0; i < nrows; i++) {
	  for (int j = 0; j < ncols; j++, k++) {
	    NumPyType<ToElem>::toCPP(rdata[k], dst[CVD::ImageRef(j, i)]);
	  }
	}
      }
      else {
	PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	if (iter == 0) {
	  throw std::string("Could not grab iterator!");
	}
	typedef typename NumPyType<ToElem>::ElemType ElemType;
	for (int i = 0; iter->index < iter->size && i < nrows; i++) {
	  for (int j = 0; j < ncols; i++) {
	    ElemType src = *(ElemType*)iter->dataptr;
	    NumPyType<ToElem>::toCPP(src, dst[CVD::ImageRef(j, i)]);
	    PyArray_ITER_NEXT(iter);
	  }
	}
      }
    }

    // This version involves a copy of an array.
    static CVD::Image<ToElem> convert(PyArrayObject *src) {
      CVD::Image<ToElem> retval(cols(src), rows(src));
      convert(src, retval);
      return retval;
    }
  };
    
  template <>
  template <typename ToElem>
  struct Converter<PyArrayObject*, CVD::BasicImage<ToElem> > {

    static void convert(PyArrayObject *src, CVD::BasicImage<ToElem> &dst) {
      if (PyArray_NDIM(src) != 2) {
	throw std::string("The NumPy array passed must be two-dimensional!");
      }
      if (PyArray_TYPE(src) != NumPyType<ToElem>::type) {
	throw std::string("The NumPy array passed must be of type "
			  + NumPyType<ToElem>::name()
			  + " (type code " + NumPyType<ToElem>::code() + ")!");
      }
      int nrows = src->dimensions[0];
      int ncols = src->dimensions[1];
      if (dst.data() == 0 && NumPyType<ToElem>::type == NPY_OBJECT && !NumPyType<ToElem>::isCPPArrayOfPyObjects()) {
	throw std::string("CVD::BasicImage cannot use a NumPy PyObject array buffer as its memory buffer unless CVD::BasicImage<T==PyObject*>!");
      }
      else if (dst.data() == 0) {
	/** We're using the same underlying buffer as a NumPy
	    array. Since, complicated striding is not yet supported in
	    TooN, we must required contiguity for the moment.*/
	if (!PyArray_ISCONTIGUOUS(src)) {
	  throw std::string("The NumPy array passed must be contiguous!");
	}
	dst = CVD::BasicImage<ToElem>((ToElem*)src->data, CVD::ImageRef(ncols, nrows));
      }
      else {
	if (dst.size().y != nrows && dst.size().x != ncols) {
	  throw std::string("Image size mismatch!");
	}
	if (PyArray_ISCONTIGUOUS(src)) {
	  typedef typename NumPyType<ToElem>::ElemType ElemType;
	  ElemType *rdata = (ElemType*)src->data;
	  for (int y = 0, k = 0; y < nrows; y++) {
	    for (int x = 0; x < ncols; x++, k++) {
	      NumPyType<ToElem>::toCPP(rdata[k], dst[y][x]);
	    }
	  }
	}
	else {
	  PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	  if (iter == 0) {
	    throw std::string("Could not grab iterator!");
	  }
	  typedef typename NumPyType<ToElem>::ElemType ElemType;
	  for (int i = 0; iter->index < iter->size && i < nrows; i++) {
	    for (int j = 0; j < ncols; i++) {
	      NumPyType<ToElem>::toCPP(*((ElemType*)iter->dataptr), dst[CVD::ImageRef(j, i)]);
	      PyArray_ITER_NEXT(iter);
	    }
	  }  
	}
      }
    }

    static CVD::BasicImage<ToElem> convert(PyArrayObject *src) {
      CVD::BasicImage<ToElem> retval((ToElem*)0, CVD::ImageRef(cols(src), rows(src)));
      convert(src, retval);
      return retval;
    }

  };

  template <typename Elem>
  class BiAllocator<CVD::BasicImage<Elem>, PyArrayObject*, Elem, std::pair<int, int> > {

  public:
    BiAllocator(const std::pair<int, int> &sz) : sz(sz), second(0) {
      npy_intp dims[] = {sz.second, sz.first};
      second = (PyArrayObject*)PyArray_SimpleNew(2, dims, NumPyType<Elem>::type);
      if (second == 0) {
	throw std::string("MultiAllocator: error when allocating new NumPy array.");
      }
    }
  
    CVD::BasicImage<Elem> getFirst() {
      return CVD::BasicImage<Elem>((Elem*)second->data, CVD::ImageRef(sz.first, sz.second));
    }

    PyArrayObject *getSecond() {
      return second;
    }

  ///   Type2 &getSecond();
  ///   const Type1 &getFirst() const;
  ///   const Type2 &getSecond() const;

  private:
    const std::pair<size_t, size_t> sz;

    PyArrayObject *second;
  };

}

#endif
