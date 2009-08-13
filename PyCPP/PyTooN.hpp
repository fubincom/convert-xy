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

/// \file   PyTooN.hpp
///
/// \brief This header contains converter classes for converting between
/// TooN matrices/vectors and NumPy arrays.
///
/// \author Damian Eads

#ifndef SCTL_PYTOON_HPP
#define SCTL_PYTOON_HPP

#include "converter.hpp"
#include "NumPyTypes.hpp"
#include "TooN/TooN.h"

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace PyCPP {

  /// A template specialization class for copying the contents of a
  /// one-dimensional NumPy array into a TooN::Vector.
  ///
  /// \tparam Size   The size of the TooN::Vector, which can be either
  ///                static or dynamic.
  ///
  /// \tparam ToElem The type of the elements stored in the
  ///                TooN::Vector.
  ///
  /// \tparam VBase  The storage specifier.
  template <>
  template <int Size, typename ToElem, typename VBase>
  struct Converter<PyArrayObject*, TooN::Vector<Size, ToElem, VBase> > {

    /// Copies the contents of a NumPy array into a TooN::Vector. The
    /// Vector of a size matching the size of the NumPy array must be
    /// allocated beforehand.
    ///
    /// The TooN::Vector can be preallocated before conversion as the
    /// following example illustrates. The <code>from</code> variable
    /// is a PyArrayObject*.
    ///
    /// \code
    /// int n = len(from);
    /// typedef Converter<PyArrayObject*, TooN::Vector<TooN::Dynamic, double> > converter;
    /// TooN::Vector<TooN::Dynamic, double> to(n);
    /// converter::convert(from, to);
    /// \endcode
    ///
    /// @param src  A one-dimensional NumPy array to convert.
    /// @param dst  A preallocated TooN::Vector.
    ///
    /// \exception std::string Thrown if the array passed is neither
    ///                        one dimensional, of the right size, or
    ///                        of a compatible type.
    static void convert(PyArrayObject *src, TooN::Vector<Size, ToElem, VBase> &dst) {
      if (PyArray_NDIM(src) != 1) {
	throw std::string("PyCPP: The NumPy array passed must be one-dimensional!");
      }
      if (PyArray_TYPE(src) != NumPyType<ToElem>::type) {
	throw std::string("PyCPP: The NumPy array passed must be of type "
			  + NumPyType<ToElem>::name()
			  + " (type code " + NumPyType<ToElem>::code() + ")!");
      }
      int sz = src->dimensions[0];
      if (dst.size() != sz) {
	throw std::string("PyCPP: Expected vector of the right size.");
      }
      if (PyArray_ISCONTIGUOUS(src)) {
	typedef typename NumPyType<ToElem>::ElemType ElemType;
	ElemType *rdata = (ElemType*)src->data;
	for (int i = 0; i < sz; i++) {
	  NumPyType<ToElem>::toCPP(rdata[i], dst[i]);
	}
      }
      else {
	PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	if (iter == 0) {
	  throw std::string("PyCPP: Could not grab iterator!");
	}
	typedef typename NumPyType<ToElem>::ElemType ElemType;
	for (int i = 0; iter->index < iter->size; i++) {
	  NumPyType<ToElem>::toCPP(*((ElemType*)iter->dataptr), dst[i]);
	  PyArray_ITER_NEXT(iter);
	}
      }
    }

    /// Copies the contents of a NumPy array into a new TooN::Vector
    /// allocated by this function. This version involves an extra
    /// copy for its return so the convert(from, to) form is often
    /// preferable.
    ///
    /// @param src  A one-dimensional NumPy array to convert.
    ///
    /// @return   A newly allocated TooN::Vector with its contents
    ///           containing a copy of the NumPy array's buffer.
    /// \exception std::string Thrown if the array passed is neither
    ///                        one dimensional, of the right size, or
    ///                        of a compatible type.
    static TooN::Vector<Size, ToElem, VBase> convert(PyArrayObject *src) {
      TooN::Vector<Size, ToElem, VBase> retval(len(src));
      convert(src, retval);
      return retval;
    }
  };


  /// A template specialization class for copying the contents of a
  /// one-dimensional NumPy array into a reference TooN::Vector (or)
  /// copying the NumPy buffer pointer so it is shared by a reference
  /// TooN::Vector.
  template <>
  template <int Size, typename ToElem>
  struct Converter<PyArrayObject*, TooN::Vector<Size, ToElem, TooN::Reference> > {
    /// This converter has two semantics, depending on how its invoked. It either
    ///
    /// \li 1. copies the contents of a NumPy array if the TooN::Vector's underlying
    /// buffer pointer is non-NULL.
    /// \li 2. copies the buffer pointer of a NumPy array into a reference TooN::Vector
    /// buffer pointer if the TooN::Vector's underlying buffer is NULL.
    ///
    /// The example below copies a NumPy array's buffer pointer into a reference TooN::Vector
    /// which is initialized to NULL.
    ///
    /// \code
    /// int n = len(from);
    /// typedef Converter<PyArrayObject*, TooN::Vector<TooN::Dynamic, double, TooN::Reference> > converter;
    /// // Initialize the reference Vector to NULL.
    /// TooN::Vector<TooN::Dynamic, double, Reference> to(0, n);
    ///
    /// // Copy the buffer pointer.
    /// converter::convert(from, to);
    /// \endcode
    ///
    /// @param src  A one-dimensional NumPy array to convert.
    /// @param dst  A preallocated TooN::Vector.
    ///
    /// \exception std::string Thrown if the array passed is neither
    ///                        one dimensional, of the right size, or
    ///                        of a compatible type.
    static void convert(PyArrayObject *src, TooN::Vector<Size, ToElem, TooN::Reference> &dst) {
      if (PyArray_NDIM(src) != 1) {
	throw std::string("PyCPP: The NumPy array passed must be one-dimensional!");
      }
      if (PyArray_TYPE(src) != NumPyType<ToElem>::type) {
	throw std::string("PyCPP: The NumPy array passed must be of type "
			  + NumPyType<ToElem>::name()
			  + " (type code " + NumPyType<ToElem>::code() + ")!");
      }
      int sz = src->dimensions[0];
      if (dst.size() != sz) {
	throw std::string("PyCPP: Expected vector of static size.");
      }
      if (dst.my_data == 0) {
	// When TooN::Reference is specified, we're using the same underlying buffer as a
	// NumPy array. Since, complicated striding is not yet supported in TooN, we must
	// required contiguity for the moment.
	if (!PyArray_ISCONTIGUOUS(src)) {
	  throw std::string("PyCPP: The NumPy array passed must be contiguous!");
	}
	//// FIXME FIXME FIXME
	ToElem **toon_buffer;
	toon_buffer = const_cast<ToElem**>(&dst.my_data);
	*toon_buffer = (ToElem*)src->data;
	//throw std::string("Changing TooN::Reference array underlying buffers after construction is presently unsupported!");
      }
      else {
	PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	if (iter == 0) {
	  throw std::string("PyCPP: Could not grab iterator!");
	}
	typedef typename NumPyType<ToElem>::ElemType ElemType;
	for (int i = 0; iter->index < iter->size; i++) {
	  NumPyType<ToElem>::toCPP(*((ElemType*)iter->dataptr), dst[i]);
	  PyArray_ITER_NEXT(iter);
	}
      }
    }

    static TooN::Vector<Size, ToElem, TooN::Reference> convert(PyArrayObject *src) {
      if (PyArray_NDIM(src) != 1) {
	throw std::string("PyCPP: The NumPy array passed must be one-dimensional!");
      }
      if (PyArray_TYPE(src) != NumPyType<ToElem>::type) {
	throw std::string("PyCPP: The NumPy array passed must be of type "
			  + NumPyType<ToElem>::name()
			  + " (type code " + NumPyType<ToElem>::code() + ")!");
      }
      /** When TooN::Reference is specified, we're using the same underlying buffer as a
	  NumPy array. Since, complicated striding is not yet supported in TooN, we must
	  required contiguity for the moment.*/
      if (!PyArray_ISCONTIGUOUS(src)) {
	throw std::string("PyCPP: The NumPy array passed must be contiguous!");
      }
      if (NumPyType<ToElem>::type == NPY_OBJECT && !NumPyType<ToElem>::isCPPArrayOfPyObjects()) {
	throw std::string("PyCPP: New TooN::Reference Vectors cannot use a NumPy PyObject array buffer as its memory buffer unless the Vector<*, Precision==PyObject*, Reference>!");
      }
      TooN::Vector<Size, ToElem, TooN::Reference> retval((ToElem*)src->data, len(src));
      return retval;
    }

  };


  template <>
  template <int Size, typename FromElem, typename Base>
  struct Converter<TooN::Vector<Size, FromElem, Base>, PyArrayObject*> {
    static void convert(const TooN::Vector<Size, FromElem, Base> &src, PyArrayObject *&dst) {
      npy_intp dims[] = {src.size()};
      dst = (PyArrayObject*)PyArray_SimpleNew(1, dims, NumPyType<FromElem>::type);
      if (dst == 0) {
	throw std::string("PyCPP: Error when trying to allocate new NumPy array.");
      }
      /** The NumPyElementConverterToPython typedef will enable us to either define a
	  primitive pointer (like an int*) or an object pointer
	  (PyObject**).src

	  Note: need to check whether object arrays are stored as
	  PyObject* or PyObject**'s*/    
      typedef typename NumPyType<FromElem>::ElemType NumPyElemType;
      NumPyElemType *rdata = (NumPyElemType*)dst->data;
      //(typename NumPyElementConverterToPython<FromElem>::typename ToType*)retval->data;
      for (int i = 0; i < src.size(); i++) {
	NumPyType<FromElem>::toNumPy(src[i], rdata[i]);
      }
    }
  };

  ///////////////////////////////////////////////////////////////////
  /////// MATRICES
  ///////////////////////////////////////////////////////////////////


  template <>
  template <int RSize, int CSize, typename ToElem, typename MBase>
  struct Converter<PyArrayObject*, TooN::Matrix<RSize, CSize, ToElem, MBase> > {

    static void convert(PyArrayObject *src, TooN::Matrix<RSize, CSize, ToElem, MBase> &dst) {
      if (PyArray_NDIM(src) != 2) {
	throw std::string("PyCPP: The NumPy array passed must be two-dimensional!");
      }
      if (PyArray_TYPE(src) != NumPyType<ToElem>::type) {
	throw std::string("PyCPP: The NumPy array passed must be of type "
			  + NumPyType<ToElem>::name()
			  + " (type code " + NumPyType<ToElem>::code() + ")!");
      }
      int rows = src->dimensions[0];
      int cols = src->dimensions[1];
      if (dst.num_rows() != rows && dst.num_cols() != cols) {
	throw std::string("PyCPP: Expected matrix of the right size.");
      }
      if (PyArray_ISCONTIGUOUS(src)) {
	typedef typename NumPyType<ToElem>::ElemType NumPyElemType;
	NumPyElemType *rdata = (NumPyElemType*)src->data;
	for (int i = 0, k = 0; i < rows; i++) {
	  for (int j = 0; j < cols; j++, k++) {
	    NumPyType<ToElem>::toCPP(rdata[k], dst(i, j));
	  }
	}
      }
      else {
	PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	if (iter == 0) {
	  throw std::string("PyCPP: Could not grab iterator!");
	}
	typedef typename NumPyType<ToElem>::ElemType NumPyElemType;
	for (int i = 0; i < rows; i++) {
	  for (int j = 0; j < cols && iter->index < iter->size; j++) {
	    NumPyType<ToElem>::toCPP(*((NumPyElemType*)iter->dataptr), dst(i, j));
	    PyArray_ITER_NEXT(iter);
	  }
	}
      }
    }

    // This version involves a copy of an array.
    static TooN::Matrix<RSize, CSize, ToElem, MBase> convert(PyArrayObject *src) {
      TooN::Matrix<RSize, CSize, ToElem, MBase> retval(rows(src), cols(src));
      convert(src, retval);
      return retval;
    }
  };

  template <>
  template <int RSize, int CSize, typename ToElem>
  struct Converter<PyArrayObject*, TooN::Matrix<RSize, CSize, ToElem, TooN::Reference::RowMajor> > {

    static void convert(PyArrayObject *src, TooN::Matrix<RSize, CSize, ToElem, TooN::Reference::RowMajor> &dst) {
      if (PyArray_NDIM(src) != 2) {
	throw std::string("PyCPP: The NumPy array passed must be two-dimensional!");
      }
      if (PyArray_TYPE(src) != NumPyType<ToElem>::type) {
	throw std::string("PyCPP: The NumPy array passed must be of type "
			  + NumPyType<ToElem>::name()
			  + " (type code " + NumPyType<ToElem>::code() + ")!");
      }
      int nrows = src->dimensions[0];
      int ncols = src->dimensions[1];
      if (dst.num_rows() != nrows && dst.num_cols() != ncols) {
	throw std::string("PyCPP: Expected vector of static size.");
      }
      if (dst.my_data == 0) {
	/** When TooN::Reference is specified, we're using the same underlying buffer as a
	    NumPy array. Since, complicated striding is not yet supported in TooN, we must
	    required contiguity for the moment.*/
	if (!PyArray_ISCONTIGUOUS(src)) {
	  throw std::string("PyCPP: The NumPy array passed must be contiguous!");
	}
	/// FIXME FIXME FIXME
	/// dst.my_data = (ToElem*)src->data;
	ToElem **toon_buffer;
	toon_buffer = const_cast<ToElem**>(&dst.my_data);
	*toon_buffer = (ToElem*)src->data;

	//throw std::string("Changing TooN::Reference array underlying buffers after construction is presently unsupported!");
      }
      else {
	PyArrayIterObject* iter = (PyArrayIterObject*)PyArray_IterNew((PyObject*)src);
	if (iter == 0) {
	  throw std::string("PyCPP: Could not grab iterator!");
	}
	typedef typename NumPyType<ToElem>::ElemType NumPyElemType;
	for (int i = 0; iter->index < iter->size; i++) {
	  for (int j = 0; j < ncols; j++) {
	    NumPyType<ToElem>::toCPP(*((NumPyElemType*)iter->dataptr), dst(i, j));
	    PyArray_ITER_NEXT(iter);
	  }
	}
      }
    }

    static TooN::Matrix<RSize, CSize, ToElem, TooN::Reference::RowMajor> convert(PyArrayObject *src) {

      if (PyArray_NDIM(src) != 2) {
	throw std::string("PyCPP: The NumPy array passed must be two-dimensional!");
      }
      if (PyArray_TYPE(src) != NumPyType<ToElem>::type) {
	throw std::string("PyCPP: The NumPy array passed must be of type "
			  + NumPyType<ToElem>::name()
			  + " (type code " + NumPyType<ToElem>::code() + ")!");
      }
      int nrows = src->dimensions[0];
      int ncols = src->dimensions[1];
      /** When TooN::Reference is specified, we're using the same underlying buffer as a
	  NumPy array. Since, complicated striding is not yet supported in TooN, we must
	  required contiguity for the moment.*/
      if (!PyArray_ISCONTIGUOUS(src)) {
	throw std::string("PyCPP: The NumPy array passed must be contiguous!");
      }
      if (NumPyType<ToElem>::type == NPY_OBJECT && !NumPyType<ToElem>::isCPPArrayOfPyObjects()) {
	throw std::string("PyCPP: New TooN::Reference Matrix objects cannot use a NumPy PyObject array buffer as its memory buffer unless Matrix<*, Precision==PyObject*, Reference>!");
      }
      return TooN::Matrix<RSize, CSize, ToElem, TooN::Reference::RowMajor>((ToElem*)src->data, nrows, ncols);
    }

  };

  template <>
  template <int RSize, int CSize, typename FromElem, typename Base>
  struct Converter<TooN::Matrix<RSize, CSize, FromElem, Base>, PyArrayObject*> {
    static void convert(const TooN::Matrix<RSize, CSize, FromElem, Base> &src, PyArrayObject *&dst) {
      npy_intp a[] = {src.num_cols(), src.num_rows()};
      dst = (PyArrayObject*)PyArray_SimpleNew(2, a, NumPyType<FromElem>::type);
      if (dst == 0) {
	throw std::string("PyCPP: Error when trying to allocate new NumPy array.");
      }
      typedef typename NumPyType<FromElem>::ElemType NumPyElemType;
      NumPyElemType *rdata = (NumPyElemType*)dst->data;
      for (int i = 0; i < src.num_rows(); i++) {
	for (int j = 0; j < src.num_cols(); j++, rdata++) {
	  NumPyType<FromElem>::toNumPy(src(i, j), rdata[i]);
	}
      }
    }
  };

  template <typename Elem>
  template <int RSize, int CSize>
  class BiAllocator<TooN::Matrix<RSize, CSize, Elem, TooN::Reference::RowMajor>, PyArrayObject*, Elem, std::pair<int, int> > {

    BiAllocator(const std::pair<int, int> &sz) : sz(sz), second(0) {
      npy_intp dims[] = {sz.first, sz.second};
      second = (PyArrayObject*)PyArray_SimpleNew(2, dims, NumPyType<Elem>::type);
      if (second == 0) {
	throw std::string("MultiAllocator: error when allocating new NumPy array.");
      }
    }
  
    TooN::Matrix<RSize, CSize, Elem, TooN::Reference::RowMajor> &getFirst() {
      return TooN::Matrix<RSize, CSize, Elem, TooN::Reference::RowMajor>((Elem*)second->data);
    }

    const TooN::Matrix<RSize, CSize, Elem, TooN::Reference::RowMajor> &getFirst() const {
      return TooN::Matrix<RSize, CSize, Elem, TooN::Reference::RowMajor>((Elem*)second->data);
    }

    PyArrayObject *getSecond() {
      return second;
    }

  ///   Type2 &getSecond();
  ///   const Type1 &getFirst() const;
  ///   const Type2 &getSecond() const;

  private:
    const size_t sz;
    TooN::Matrix<RSize, CSize, Elem, TooN::Reference::RowMajor> first;

    PyArrayObject *second;
  };

  template <typename Elem>
  template <int Size>
  class BiAllocator<TooN::Vector<Size, Elem, TooN::Reference>, PyArrayObject*, Elem, std::pair<int, int> > {

    BiAllocator(const std::pair<int, int> &sz) : sz(sz), second(0) {
      npy_intp dims[] = {sz.first, sz.second};
      second = (PyArrayObject*)PyArray_SimpleNew(2, dims, NumPyType<Elem>::type);
      if (second == 0) {
	throw std::string("MultiAllocator: error when allocating new NumPy array.");
      }
    }
  
    TooN::Vector<Size, Elem, TooN::Reference> &getFirst() {
      return TooN::Vector<Size, Elem, TooN::Reference>((Elem*)second->data);
    }

    const TooN::Vector<Size, Elem, TooN::Reference> &getFirst() const {
      return TooN::Vector<Size, Elem, TooN::Reference>((Elem*)second->data);
    }

    PyArrayObject *getSecond() {
      return second;
    }

  ///   Type2 &getSecond();
  ///   const Type1 &getFirst() const;
  ///   const Type2 &getSecond() const;

  private:
    const size_t sz;
    TooN::Vector<Size, Elem, TooN::Reference> first;

    PyArrayObject *second;
  };


}

#endif
