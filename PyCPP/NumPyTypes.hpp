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

#ifndef SCTL_NUMPYTYPES_HPP
#define SCTL_NUMPYTYPES_HPP

/// \file   NumPyTypes.hpp
///
/// \brief  This header contains utility classes and functions for
/// NumPy array type interpretation at run time and compile time.
///
/// \author Damian Eads

#include "converter.hpp"
#include "scalars.hpp"
#include <Python.h>
#include <numpy/arrayobject.h>

#include <string>

/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.

namespace PyCPP {

  /// A very general templated class for encapsulating type
  /// information about NumPy arrays with specific element types. This
  /// class can be used for simple type checking of the elements in a
  /// NumPy array buffer. The template argument CPP is the type of a
  /// desired element in C++. From this, we can obtain the
  /// corresponding NumPy type code and type letter.
  ///
  /// For example, to return the NumPy array buffer as a double *
  /// only if double's are actually stored in the buffer, we can do
  /// the following:
  ///
  /// \code
  ///   double *getBuffer(PyArrayObject *obj) {
  ///      double *retval(0);
  ///      if (NumPyType<double>::isCompatible(src)) {
  ///         retval = src->data;
  ///      }
  ///      else {
  ///         throw std::string("Expected array of type double!");
  ///      }
  ///      return retval;
  ///   }
  /// \endcode
  ///
  /// Or equivalently, we can compare the NumPy type code directly
  /// as follows
  /// \code
  ///   double *getBuffer(PyArrayObject *obj) {
  ///      double *retval(0);
  ///      if (NumPyType<double>::type == PyArray_TYPE(src)) {
  ///         retval = src->data;
  ///      }
  ///      else {
  ///         throw std::string("Expected array of type double!");
  ///      }
  ///      return retval;
  ///   }
  /// \endcode
  ///
  /// If the NumPy array stored strings, dictionaries, or other NumPy
  /// arrays, the underlying buffer contains pointers to PyObject. We
  /// can check for whether an array contains buffers
  ///
  /// \tparam CPP The type of the CPP object of interest.
  template<class CPP> struct NumPyType {

    /// The NumPy integer type code corresponding to objects of type
    /// CPP. By default, this is NPY_OBJECT. This is overridden by
    /// template specialization for each primitive numeric type in
    /// C++.
    static const int   type = NPY_OBJECT;

    /// The name of the CPP type.
    static std::string name(){ return "void *"; }

    /// The NumPy character type code. By default this is 'O'.
    static char code() { return 'O'; }

    /// The type of the elements stored in the NumPy array's
    /// buffer. It's either a primitive or an array of PyObject*.
    static typedef PyObject* ElemType;

    /// The default case assumes the NumPy array elements are Python
    /// objects, not NumPy scalars. As such, they're converted to
    /// a type C by invoking a Converter<PyObject*, C>::convert function.
    ///
    /// @param src     The object to convert from.
    /// @param dst     The object to convert to.
    static void toCPP(PyObject *src, CPP &dst) {
      Converter<PyObject*, CPP>::convert(src, dst);
    }

    /// The default case assumes the NumPy array elements are Python
    /// objects, not NumPy scalars.
    ///
    /// @param src     The object to convert from.
    ///
    /// @return The converted element.
    static CPP toCPP(PyObject *src) {
      return Converter<PyObject*, CPP>::convert(src);
    }

    /// The default case assumes the NumPy array elements are Python
    /// objects, not NumPy scalars. As such, the CPP object is converted
    /// to a Python object by invoking the templated conversion function.
    ///
    /// @param src     The CPP object to convert from.
    /// @param dst     A pointer to a PyObject* which will be assigned
    ///                after the converted Python object is created.
    static void toNumPy(const CPP &src, PyObject *&dst) {
      Converter<CPP, PyObject*>::convert(src, dst);
    }


    /// The default case assumes the NumPy array elements are Python
    /// objects, not NumPy scalars. As such, the CPP object is converted
    /// to a Python object by invoking the templated conversion function.
    ///
    /// @param src     The CPP object to convert from.
    ///
    /// @return  A pointer to the converted element as a Python object.

    static PyObject* toNumPy(const CPP &src) {
      return Converter<CPP, PyObject*>::convert(src);
    }

    /// Returns True if the NumPy array passed contains elements with
    /// a type that is compatible with the CPP type. Note that the
    /// types may be compatible even though the NumPy element type is
    /// different from the CPP type. For example, a NumPy array of
    /// strings will contain PyObject's but the CPP type is
    /// std::string. Conversion still needs to take place in such
    /// situations.
    ///
    /// @param obj   The array to check for compatibility.
    ///
    /// @return True only if the NumPy array passed contains types
    /// which are compatible with the CPP type.
    static bool isCompatible(const PyArrayObject *obj) {
      return PyArray_TYPE(obj) == type;
    }

    /// Returns True if the NumPy arrays containing elements
    /// compatible with type CPP are stored as PyObjects.
    static bool isCPPArrayOfPyObjects() { return false; }
  };

  /// This class encapsulates type information when no conversion is
  /// performed on NumPy PyObject array elements. This is useful
  /// for converting NumPy arrays to C++ arrays or vectors of type
  /// PyObject*.
  template<>
  struct NumPyType<PyObject*>
  {

    /// The NumPy type code PyObject*.
    static const int   type = NPY_OBJECT;
    static std::string name(){ return "PyObject*"; }
    static char code() { return 'O'; }
    static typedef PyObject* ElemType;

    static void toCPP(PyObject *src, PyObject *&dst) {
      // The default case assumes the NumPy array elements are Python
      // objects, not NumPy scalars. As such, they're converted to
      // a type C by invoking a Converter<PyObject*, C>::convert function.
      dst = src;
    }

    static void toNumPy(PyObject *&src, PyObject *&dst) { dst = src; }

    static bool isCPPArrayOfPyObjects() { return true; }
  };

#define DEF_NUMPY_TYPE_CORRESPONDENCE(Type, PyType)	\
  /** Encapsulates type information for NumPy arrays of a specific NumPy buffer element type and corresponding C type. */ \
  template<> struct NumPyType<Type>			\
  {							\
    /** The NumPy type code. */                         \
    static const int   type = PyType;			\
    /** The name of the corresponding C type as a string, which is useful for error reporting. */  \
    static std::string name(){ return #Type;}		\
    /** The name of the corresponding NumPy dtype code as a character, which is useful for error reporting. */  \
    static char code(){ return PyType##LTR;}		\
    /** The type of each element stored in the NumPy array buffer. */  \
    static typedef Type ElemType;			\
    /** Converts a NumPy array element to a CPP array element. For simple primitive types (not PyObject*, e.g. int, float), these types are the same. */  \
    static void toCPP(const Type &src, Type &dst) {dst = src;}	\
    /** Converts a CPP array element to a NumPy array element. For simple primitive types (not PyObject*, e.g. int, float), these types are the same so a simple assignment occurs. */  \
    static void toNumPy(const Type &src, Type &dst) {dst = src;} \
    /** Returns True if the NumPy array corresponding to this type code contains a buffer of PyObject*'s. */ \
    static bool isCPPArrayOfPyObjects() { return false; } \
  }

#define DOXYGEN_EXPAND_MACRO
#if defined(DOXYGEN_EXPAND_MACRO)
  DEF_NUMPY_TYPE_CORRESPONDENCE(bool          , NPY_BOOL );
  DEF_NUMPY_TYPE_CORRESPONDENCE(unsigned char , NPY_UBYTE );
  DEF_NUMPY_TYPE_CORRESPONDENCE(char          , NPY_BYTE );
  DEF_NUMPY_TYPE_CORRESPONDENCE(unsigned short, NPY_USHORT);
  DEF_NUMPY_TYPE_CORRESPONDENCE(short         , NPY_SHORT );
  DEF_NUMPY_TYPE_CORRESPONDENCE(unsigned int  , NPY_UINT  );
  DEF_NUMPY_TYPE_CORRESPONDENCE(int           , NPY_INT   );
  DEF_NUMPY_TYPE_CORRESPONDENCE(float         , NPY_FLOAT );
  DEF_NUMPY_TYPE_CORRESPONDENCE(double        , NPY_DOUBLE);
  DEF_NUMPY_TYPE_CORRESPONDENCE(unsigned long , NPY_ULONG);
  DEF_NUMPY_TYPE_CORRESPONDENCE(long,           NPY_LONG);
  DEF_NUMPY_TYPE_CORRESPONDENCE(long long,      NPY_LONGLONG);
  DEF_NUMPY_TYPE_CORRESPONDENCE(unsigned long long,      NPY_ULONGLONG);
  DEF_NUMPY_TYPE_CORRESPONDENCE(long double   , NPY_LONGDOUBLE);
#endif

  /// A templated specialization for finding the dimensions, length,
  /// number of rows, and number of columns for NumPy arrays.
  template <>
  struct ContainerLength<PyArrayObject*> {

    /// Returns the length, or first dimension, of a NumPy array.
    ///
    /// @param src     A NumPy array.
    ///
    /// @return        The number of dimensions of the NumPy array passed.
    static int len(PyArrayObject *src) {
      if (ndim(src) == 0) {
	throw std::string("PyCPP: len() of unsized object");
      }
      return PyArray_DIM(src, 0);
    }

    /// Return the length, or first dimension, of a Python object
    /// assumed to be a NumPy array.
    ///
    /// @param src     A NumPy array, passed as a <code>PyObject*</code>.
    ///
    /// @return   The length of the NumPy array passed.
    ///
    /// \exception std::string Thrown if the object passed is not a
    ///                        NumPy array.
    static int len(PyObject *src) {
      if (!PyArray_Check(src)) {
	throw std::string("PyCPP: len() on object expected to be NumPy array.");
      }
      if (ndim(src) == 0) {
	throw std::string("PyCPP: len() of unsized object");
      }
      return PyArray_DIM(src, 0);
    }

    /// Returns the number of dimensions of a NumPy array.
    ///
    /// @param src     A NumPy array.
    ///
    /// @return        The number of dimensions of the NumPy array passed.
    static int ndim(PyArrayObject *src) {
      return PyArray_NDIM(src);
    }

    /// Returns the number of dimensions of a Python object, assumed
    /// to be a NumPy array.
    ///
    /// @param src     A NumPy array passed as a <code>PyObject*</code>.
    ///
    /// @return        The number of dimensions of the NumPy array passed.
    ///
    /// \exception std::string Thrown if the object passed is not a
    ///                        NumPy array.
    static int ndim(PyObject *src) {
      if (!PyArray_Check(src)) {
	throw std::string("PyCPP: ndim() on object expected to be NumPy array.");
      }
      return PyArray_NDIM(src);
    }

    /// Returns the size along a specified dimension of a NumPy array.
    ///
    /// @param src     A NumPy array.
    /// @param i       The dimension of interest.
    ///
    /// @return        The size of the dimension specified.
    ///
    /// \exception std::string Thrown if the dimension is out of bounds.
    static int dim(PyArrayObject *src, int i) {
      if (i >= ndim(src) || i < 0) {
	throw std::string("PyCPP: dimension specifier out of bounds");
      }
      return PyArray_DIM(src, i);
    }

    /// Returns the size along a specified dimension of a Python object
    /// assumed to be a NumPy array.
    ///
    /// @param src     A NumPy array, passed as a <code>PyObject*</code>.
    /// @param i       The dimension of interest.
    ///
    /// @return        The size of the dimension specified.
    ///
    /// \exception std::string Thrown if the object passed is not a
    ///                        NumPy array.
    static int dim(PyObject *src, int i) {
      if (!PyArray_Check(src)) {
	throw std::string("PyCPP: dim() on object expected to be NumPy array.");
      }
      if (i >= ndim(src) || i < 0) {
	throw std::string("PyCPP: dimension specifier out of bounds");
      }
      return PyArray_DIM(src, i);
    }

    /// Returns the number of rows of a two-dimensional Python sequence
    /// type such as a NumPy array.
    ///
    /// @param from              A two-dimensional Python sequence.
    ///
    /// @return The number of rows.
    ///
    /// \exception std::string Thrown if the array is not two-dimensional.
    static int rows(PyArrayObject *src) {
      if (ndim((PyArrayObject*)src) < 2) {
	throw std::string("PyCPP: rows() of an array that's not two-dimensional");
      }
      return PyArray_DIM(src, 0);
    }


    /// Returns the number of rows of a two-dimensional Python sequence
    /// type such as a NumPy array.
    ///
    /// @param from              A two-dimensional Python sequence.
    ///
    /// @return The number of rows.
    ///
    /// \exception std::string Thrown if the array is not two-dimensional or
    ///                        if the object passed is not an array.
    static int rows(PyObject *src) {
      if (!PyArray_Check(src)) {
	throw std::string("PyCPP: rows() on object expected to be NumPy array.");
      }
      if (ndim((PyArrayObject*)src) < 2) {
	throw std::string("PyCPP: rows() of an array that's not two-dimensional");
      }
      return PyArray_DIM(src, 0);
    }


    /// Returns the number of columns of a two-dimensional Python
    /// sequence type such as a NumPy array.
    ///
    /// @param from              A two-dimensional Python sequence.
    ///
    /// @return The number of rows.
    ///
    /// \exception std::string Thrown if the array is not two-dimensional.
    static int cols(PyArrayObject *src) {
      if (ndim((PyArrayObject*)src) < 2) {
	throw std::string("PyCPP: cols() of an array that's not two-dimensional");
      }
      return PyArray_DIM(src, 1);
    }

    /// Returns the number of columns of a two-dimensional Python sequence
    /// type such as a NumPy array.
    ///
    /// @param from              A two-dimensional Python sequence.
    ///
    /// @return The number of columns.
    ///
    /// \exception std::string Thrown if the array is not two-dimensional or
    ///                        if the object passed is not an array.
    static int cols(PyObject *src) {
      if (!PyArray_Check(src)) {
	throw std::string("PyCPP: cols() on object expected to be NumPy array.");
      }
      if (ndim((PyArrayObject*)src) < 2) {
	throw std::string("PyCPP: cols() of an array that's not two-dimensional");
      }
      return PyArray_DIM(src, 1);
    }

  };
}

#endif
