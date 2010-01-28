// Convert-XY: A library for interchanging C++ and Python objects
//
// Copyright (C) 2009-2010 Damian Eads
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// As a special exception, you may use this file as part of a free software
// library without restriction.  Specifically, if other files instantiate
// templates or use macros or inline functions from this file, or you compile
// this file and link it with other files to produce an executable, this
// file does not by itself cause the resulting executable to be covered by
// the GNU General Public License.  This exception does not however
// invalidate any other reasons why the executable file might be covered by
// the GNU General Public License.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

/// \file   CVD.hpp
///
/// \brief This header is included to convert between NumPy/Python
/// containers and LIBCVD constructs.
///
/// \author Damian Eads

#ifndef CONVERTXY_CVD_HPP
#define CONVERTXY_CVD_HPP

#include <cvd/image.h>

#include "converter.hpp"
#include "numpy_scalars.hpp"
#include "holder.hpp"

namespace ConvertXY {

  using namespace std;
  using namespace CVD;

  template <class T>
  struct Zero<BasicImage<T>, false > {
    static BasicImage<T> zero() {
      return BasicImage<T>(0, ImageRef());
    }
  };

  template <class T>
  struct ObjectFactory<BasicImage<T> > {
    static BasicImage<T> create(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      return BasicImage<T>((T*)bufferPtr, ImageRef(dimensions[1], dimensions[0]));
    }

    static BasicImage<T> create_ptr(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      return new BasicImage<T>((T*)bufferPtr, ImageRef(dimensions[1], dimensions[0]));
    }
  };

  template <class T>
  struct ObjectFactory<Image<T> > {
    static Image<T> create(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      return Image<T>(ImageRef(dimensions[1], dimensions[0]));
    }

    static Image<T> create_ptr(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      return new Image<T>(ImageRef(dimensions[1], dimensions[0]));
    }
  };

  template <class T, class BA>
  struct DefaultToCPPConvertAction<Image<T>, BA > {
    typedef AllocateCopy<> Action;
  };

  template <class T, class BA>
  struct DefaultToCPPConvertAction<Holder<BasicImage<T> >, BA > {
    typedef Reuse<> Action;
  };

  template <class T, class BA>
  struct DefaultToCPPConvertAction<BasicImage<T>, BA > {
    typedef Reuse<> Action;
  };

  template <>
  template <class T>
  struct ConvertToCPP<Image<T>, PyArrayObject*, AllocateCopy<> >
    : public ConvertToCPPBase<Image<T>, AllocateCopy<> > {

    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}

    void convert_object_array(PyObject *src, Image<T> &dst) const {
      if (PyArray_NDIM(src) == 2) {
	int ysize = PyArray_DIM(src, 0);
	int xsize = PyArray_DIM(src, 1);
	dst.resize(ImageRef(xsize, ysize));
	PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	if (iter == 0) {
	  throw Py::RuntimeError("ToCPP<Image<T>, Py::Sequence> Could not grab iterator!");
	}
	
	for (int y = 0; y < ysize; y++) {
	  for (int x = 0; iter->index < iter->size && x < xsize; x++) {
	    PyObject *item = (PyObject*)iter->dataptr;
	    Py::Object _item(item);
	    ConvertToCPPBase<T, Copy<> >
	      &converter(ToCPPDispatch<T, Copy<> >::getConverter(item));
	    converter.convert(item, dst[y][x]);
	    PyArray_ITER_NEXT(iter);
	  }
	}      
      }
      else {
	throw Py::RuntimeError("ToCPP<Image<T>, Py::Sequence>: NumPy array must be two dimensional!");
      }
    }
    
    template <class WalkableTypeList>
    void walk_type_list(PyObject *src, Image<T> &dst) const {
      typedef typename WalkableTypeList::Head HeadType;
      if (PyArray_TYPE(src) == NumPyType<HeadType>::numpy_code) {
	if (PyArray_NDIM(src) == 2) {
	  int ysize = PyArray_DIM(src, 0);
	  int xsize = PyArray_DIM(src, 1);
	  dst.resize(ImageRef(xsize, ysize));
	  PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	  if (iter == 0) {
	    throw Py::RuntimeError("ToCPP<Image<T>, NumPyArray> Could not grab iterator!");
	  }

	  for (int y = 0; y < ysize; y++) {
	    for (int x = 0; iter->index < iter->size && x < xsize; x++) {
	      HeadType *item = (HeadType*)iter->dataptr;
	      dst[y][x] = *item;
	      PyArray_ITER_NEXT(iter);
	    }
	  }
	}
	else {
	  throw Py::RuntimeError("ToCPP<Image<T>, NumPyArray>: NumPy array must be two-dimensional.");
	}
      }
      else {
	typedef typename WalkableTypeList::Tail TailType;
	if (IsEnd<TailType>::end) {
	  throw Py::TypeError("No way to convert type");
	}
	else {
	  // to make the static type-checker happy.
	  walk_type_list<typename AdvanceIfPossible<WalkableTypeList>::choice >(src, dst);
	}
      }
    }
    
    virtual void convert(PyObject *src, Image<T> &dst) const {
      Py::Object _src(src);
      if (PyArray_TYPE(src) == NPY_OBJECT || PyArray_TYPE(src) == NPY_STRING) {
	convert_object_array(src, dst);
      }
      else if (PyArray_TYPE(src) < NPY_OBJECT) {
	walk_type_list<NumPyFloatTypes>(src, dst);
      }
      else {
	throw Py::TypeError("NumPy type code not understood!");
      }
    }

    virtual bool isImplemented() const { return true; }

    void *getBufferPointer(PyObject *obj) const {
      return PyArray_DATA(obj);
    }

    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> sz;
      Py::Sequence seq(src);
      const size_t ndims = PyArray_NDIM(src);
      for (size_t i = 0; i < ndims; i++) {
	sz.push_back(PyArray_DIM(src, i));
      }
      return sz;
    }
  };


  template <class BA>
  struct DefaultToCPPConvertAction<ImageRef, BA > {
    typedef Copy<> Action;
  };


  template <class DefaultBufferAction>
  struct DefaultToPythonStructure<ImageRef, DefaultBufferAction > {
    typedef PyList<PyInt> Structure;
  };

  template <>
  struct ConvertToCPP<ImageRef, Py::SeqBase<Py::Object>, Copy<> >
    : public ConvertToCPPBase<ImageRef, Copy<> > {
    
    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}
    
    virtual void convert(PyObject *src, ImageRef &dst) const {
      Py::Sequence seq(src);
      if (seq.size() == 1) {
	dst.y = 1;
	Py::Object x(seq[0]);
	ConvertXY::convert(x.ptr(), dst.x);
      }
      else if (seq.size() == 2) {
	Py::Object y(seq[0]);
	Py::Object x(seq[1]);
	ConvertXY::convert(x.ptr(), dst.x);
	ConvertXY::convert(y.ptr(), dst.y);
      }
      else {
	throw Py::RuntimeError("ConvertXY: There must be two elements to convert to an ImageRef");
      }
    }

    virtual bool isImplemented() const { return true; }

    void *getBufferPointer(PyObject *obj) const {
      return PyArray_DATA(obj);
    }

    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> sz;
      sz.push_back(2);
      return sz;
    }
  };

  template <>
  struct ConvertToPython<ImageRef, Copy<> > {
    static PyObject* convert(const ImageRef &src) {
      Py::List list;
      list[0] = Py::Int(src.y);
      list[1] = Py::Int(src.x);
      return Py::new_reference_to(list);
    }
  };

  template <>
  template <class T>
  struct ConvertToCPP<Image<T>, Py::SeqBase<Py::Object>, AllocateCopy<> >
    : public ConvertToCPPBase<Image<T>, AllocateCopy<> > {
    
    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}
    
    virtual void convert(PyObject *src, Image<T> &dst) const {
      Py::Sequence yseq(src);
      if (yseq.size() == 0) {
	return;
      }
      else {
	Py::Object first_element(yseq[0]);
	size_t max_size = 0;
	if (!first_element.isSequence()) {
	  dst.resize(ImageRef(yseq.size(), 1));
	  for (size_t y = 0; y < yseq.size(); y++) {
	    Py::Object pitem(yseq[y]);
	    if (pitem.isSequence()) {
	      throw Py::TypeError("ToCPP<CVD::Image<T>, Py::Seq>: inconsistent sequence.");
	    }
	    ConvertToCPPBase<T, Copy<> >
	      &converter(ToCPPDispatch<T, Copy<> >::getConverter(pitem.ptr()));
	    if (Primitive<T>::primitive) {
	      converter.convert(pitem.ptr(), dst[0][y]);
	    }
	  }
	}
	else {
	  bool all_sequences = true;
	  for (size_t y = 0; y < yseq.size(); y++) {
	    all_sequences &= yseq[y].isSequence();
	    if (all_sequences) {
	      Py::Sequence elem(yseq[y]);
	      size_t sz = elem.size();
	      if (sz > max_size) {
		max_size = sz;
		dst.resize(ImageRef(sz, yseq.size()));
	      }
	      for (size_t x = 0; x < elem.size(); x++) {
		Py::Object pitem(elem[x]);
		ConvertToCPPBase<T, Copy<> >
		  &converter(ToCPPDispatch<T, Copy<> >::getConverter(pitem.ptr()));
		if (Primitive<T>::primitive) {
		  dst[y][x] = (T)0;
		  converter.convert(pitem.ptr(), dst[y][x]);
		}
		else {
		  vector <size_t> elemDims(converter.getSize(pitem.ptr()));
		  void *elemPtr = converter.getBufferPointer(pitem.ptr());
		  dst[y][x] = ObjectFactory<T>::create(elemDims, elemPtr);
		  converter.convert(pitem.ptr(), dst[y][x]);
		}
	      }
	    }
	    else {
	      throw Py::TypeError("ToCPP<CVD::Image<T>, Py::Seq>: inconsistent sequence.");
	    }
	  }
	}
      }
    }

    virtual bool isImplemented() const { return true; }

    void *getBufferPointer(PyObject *obj) const {
      return PyArray_DATA(obj);
    }

    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> sz;
      Py::Sequence seq(src);
      const size_t ndims = PyArray_NDIM(src);
      for (size_t i = 0; i < ndims; i++) {
	sz.push_back(PyArray_DIM(src, i));
      }
      return sz;
    }
  };

  template <class ElemType, class DefaultBufferAction>
  struct DefaultToPythonStructure<Image<ElemType>, DefaultBufferAction > {
    typedef DefaultToPythonStructure<ElemType, DefaultBufferAction> ElemClass;
    typedef typename ElemClass::Structure ElemStructure;
    typedef PyArray<ElemStructure, Copy<> > Structure;
  };

  template <class ElemType, class DefaultBufferAction>
  struct DefaultToPythonStructure<BasicImage<ElemType>, DefaultBufferAction > {
    typedef DefaultToPythonStructure<ElemType, DefaultBufferAction> ElemClass;
    typedef typename ElemClass::Structure ElemStructure;
    typedef PyArray<ElemStructure, Copy<> > Structure;
  };

  template <class ElemType, class ElemStructure>
  struct ConvertToPython<Image<ElemType>, PyArray<ElemStructure, Copy<> > > {
    static PyObject* convert(const Image<ElemType> &src) {
      npy_intp dims[] = {src.size().y, src.size().x};
      typedef typename ElemStructure::CType ToElemType;
      PyObject *dst = PyArray_SimpleNew(2, dims, NumPyType<ToElemType>::numpy_code);
      if (dst == 0) {
	throw Py::RuntimeError("error when allocating numpy array!");
      }
      
      PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew(dst);
      if (iter == 0) {
	throw Py::RuntimeError("ToPython<Image<T>, NumPyArray> Could not grab iterator!");
      }

      for (int y = 0; y < src.size().y; y++) {
	for (int x = 0; iter->index < iter->size && x < src.size().x; x++) {
	  ToElemType *item = (ToElemType*)iter->dataptr;
	  *item = (ToElemType)src[y][x];
	  PyArray_ITER_NEXT(iter);
	}
      }
      return dst;
    }
  };

  template <class ElemType, class ElemStructure>
  struct ConvertToPython<BasicImage<ElemType>, PyArray<ElemStructure, Copy<> > > {
    static PyObject* convert(const BasicImage<ElemType> &src) {
      npy_intp dims[] = {src.size().y, src.size().x};
      typedef typename ElemStructure::CType ToElemType;
      PyObject *dst = PyArray_SimpleNew(2, dims, NumPyType<ToElemType>::numpy_code);
      if (dst == 0) {
	throw Py::RuntimeError("error when allocating numpy array!");
      }
      
      PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew(dst);
      if (iter == 0) {
	throw Py::RuntimeError("ToPython<BasicImage<T>, NumPyArray, Copy<> > Could not grab iterator!");
      }

      for (int y = 0; y < src.size().y; y++) {
	for (int x = 0; iter->index < iter->size && x < src.size().x; x++) {
	  ToElemType *item = (ToElemType*)iter->dataptr;
	  *item = (ToElemType)src[y][x];
	  PyArray_ITER_NEXT(iter);
	}
      }
      return dst;
    }
  };


  template <>
  template <class T>
  struct ConvertToCPP<Holder<BasicImage<T> >, PyArrayObject*, Reuse<> >
    : public ConvertToCPPBase<Holder<BasicImage<T> >, Reuse<> > {

    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}

    void convert(PyObject *src, Holder<BasicImage<T> > &dst) const {
      if (PyArray_TYPE(src) != NumPyType<T>::numpy_code) {
	throw Py::RuntimeError("ConvertXY: type unsupported for conversion to BasicImage.");
      }
      if (PyArray_ISCONTIGUOUS(src)) {
	if (PyArray_NDIM(src) == 2) {
	  dst.own(new BasicImage<T>((T*)getBufferPointer(src), CVD::ImageRef(PyArray_DIM(src, 1), PyArray_DIM(src, 0))));
	}
	else if (PyArray_NDIM(src) == 1) {
	  dst.own(new BasicImage<T>((T*)getBufferPointer(src), CVD::ImageRef(PyArray_DIM(src, 0), 1)));
	}
	else {
	  throw Py::RuntimeError("ConvertXY: array must be one or two dimensional for conversion to BasicImage.");
	}
      }
      else {
	throw Py::RuntimeError("ConvertXY: array must be contiguous for reuse conversion to BasicImage.");
      }
    }

    virtual bool isImplemented() const { return true; }

    void *getBufferPointer(PyObject *obj) const {
      return PyArray_DATA(obj);
    }

    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> sz;
      Py::Sequence seq(src);
      const size_t ndims = PyArray_NDIM(src);
      for (size_t i = 0; i < ndims; i++) {
	sz.push_back(PyArray_DIM(src, i));
      }
      return sz;
    }
  };

  template <>
  template <class T>
  struct ConvertToCPP<BasicImage<T>, PyArrayObject*, Reuse<> >
    : public ConvertToCPPBase<BasicImage<T>, Reuse<> > {

    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}

    void convert(PyObject *src, BasicImage<T> &dst) const {
      if (PyArray_TYPE(src) != NumPyType<T>::numpy_code) {
	throw Py::RuntimeError("ConvertXY: type unsupported for conversion to BasicImage.");
      }
      if (PyArray_ISCONTIGUOUS(src)) {
	if ((T*)PyArray_DATA(src) != dst.data()) {
	  throw Py::RuntimeError("ConvertXY: factory invoked incorrectly.");
	}
      }
      else {
	throw Py::RuntimeError("ConvertXY: array must be contiguous for reuse conversion to BasicImage.");
      }
    }

    virtual bool isImplemented() const { return true; }

    void *getBufferPointer(PyObject *obj) const {
      return PyArray_DATA(obj);
    }

    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> sz;
      Py::Sequence seq(src);
      const size_t ndims = PyArray_NDIM(src);
      for (size_t i = 0; i < ndims; i++) {
	sz.push_back(PyArray_DIM(src, i));
      }
      return sz;
    }
  };
}

#ifdef CONVERTXY_STL_HPP
#include "CVDSTL.hpp"
#endif

#endif
