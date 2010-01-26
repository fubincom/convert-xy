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

/// \file   numpy_scalars.hpp
///
/// \brief  This header contains converter classes for converting
/// NumPy scalars.
///
/// \author Damian Eads

// Guards
#ifndef CONVERTXY_NUMPY_SCALARS_HPP
#define CONVERTXY_NUMPY_SCALARS_HPP

#include <numpy/arrayobject.h>

#include "converter.hpp"
#include "scalars.hpp"

#ifndef CONVERTXY_NPY_ARRAY_SCALARS_H
#define CONVERTXY_NPY_ARRAY_SCALARS_H
#include <numpy/arrayscalars.h>
#endif

#include <complex>

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace ConvertXY {

  template <class CPPType, class CopyAction>
  struct DispatchPopulator<CPPType, CopyAction, 1> {
    static void populate_dispatch_map() {
      typedef ToCPPDispatch<CPPType, CopyAction> D;
      DispatchPopulator<CPPType, CopyAction, 0>::populate_dispatch_map();
      D::add(PyArray_Type, new ConvertToCPP<CPPType, PyArrayObject*, CopyAction>());
#ifdef NPY_BOOL
      D::add(PyBoolArrType_Type, new ConvertToCPP<CPPType, PyBoolScalarObject*, CopyAction>());
#endif
#ifdef NPY_UINT8
      D::add(PyUInt8ArrType_Type, new ConvertToCPP<CPPType, PyUInt8ScalarObject*, CopyAction>());
#endif
#ifdef NPY_INT8
      D::add(PyInt8ArrType_Type, new ConvertToCPP<CPPType, PyInt8ScalarObject*, CopyAction>());
#endif
#ifdef NPY_UINT16
      D::add(PyUInt16ArrType_Type, new ConvertToCPP<CPPType, PyUInt16ScalarObject*, CopyAction>());
#endif
#ifdef NPY_INT16
      D::add(PyInt16ArrType_Type, new ConvertToCPP<CPPType, PyInt16ScalarObject*, CopyAction>());
#endif
#ifdef NPY_UINT32
      D::add(PyUInt32ArrType_Type, new ConvertToCPP<CPPType, PyUInt32ScalarObject*, CopyAction>());
#endif
#ifdef NPY_INT32
      D::add(PyInt32ArrType_Type, new ConvertToCPP<CPPType, PyInt32ScalarObject*, CopyAction>());
#endif
#ifdef NPY_UINT64
      D::add(PyUInt64ArrType_Type, new ConvertToCPP<CPPType, PyUInt64ScalarObject*, CopyAction>());
#endif
#ifdef NPY_INT64
      D::add(PyInt64ArrType_Type, new ConvertToCPP<CPPType, PyInt64ScalarObject*, CopyAction>());
#endif
#ifdef NPY_UINT128
      D::add(PyUInt128ArrType_Type, new ConvertToCPP<CPPType, PyUInt128ScalarObject*, CopyAction>());
#endif
#ifdef NPY_INT128
      D::add(PyInt128ArrType_Type, new ConvertToCPP<CPPType, PyInt128ScalarObject*, CopyAction>());
#endif
#ifdef NPY_FLOAT16
      D::add(PyFloat16ArrType_Type, new ConvertToCPP<CPPType, PyFloat16ScalarObject*, CopyAction>());
      D::add(PyComplex32ArrType_Type, new ConvertToCPP<CPPType, PyComplex32ScalarObject*, CopyAction>());
#endif
#ifdef NPY_FLOAT32
      D::add(PyFloat32ArrType_Type, new ConvertToCPP<CPPType, PyFloat32ScalarObject*, CopyAction>());
      D::add(PyComplex64ArrType_Type, new ConvertToCPP<CPPType, PyComplex64ScalarObject*, CopyAction>());
#endif
#ifdef NPY_FLOAT64
      D::add(PyFloat64ArrType_Type, new ConvertToCPP<CPPType, PyFloat64ScalarObject*, CopyAction>());
      D::add(PyComplex128ArrType_Type, new ConvertToCPP<CPPType, PyComplex128ScalarObject*, CopyAction>());
#endif
#ifdef NPY_FLOAT96
      D::add(PyFloat96ArrType_Type, new ConvertToCPP<CPPType, PyFloat96ScalarObject*, CopyAction>());
      D::add(PyComplex192ArrType_Type, new ConvertToCPP<CPPType, PyComplex192ScalarObject*, CopyAction>());
#endif
#ifdef NPY_FLOAT128
      D::add(PyFloat128ArrType_Type, new ConvertToCPP<CPPType, PyFloat128ScalarObject*, CopyAction>());
      D::add(PyComplex256ArrType_Type, new ConvertToCPP<CPPType, PyComplex256ScalarObject*, CopyAction>());
#endif
#ifdef NPY_OBJECT
      D::add(PyObjectArrType_Type, new ConvertToCPP<CPPType, PyObjectScalarObject*, CopyAction>());
#endif
    }
  };

  // Boolean meta-types.

  /// A meta-type for specifying a NumPy boolean scalar for the
  /// purposes of generic meta-programming.
  class NumPyBool {
#ifdef NPY_BOOL
    public: typedef npy_bool CType;
#endif
  };

  /// A meta-type for specifying a NumPy unsigned 8-bit int scalar for
  /// the purposes of generic meta-programming.
  class NumPyUInt8 {
#ifdef NPY_UINT8
    public: typedef npy_uint8 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 8-bit int scalar for the
  /// purposes of generic meta-programming.
  class NumPyInt8 {
#ifdef NPY_INT8
    public: typedef npy_int8 CType;
#endif
  };

  /// A meta-type for specifying a NumPy unsigned 16-bit int scalar
  /// for the purposes of generic meta-programming.
  class NumPyUInt16 {
#ifdef NPY_UINT16
    public: typedef npy_uint16 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 16-bit int scalar for the
  /// purposes of generic meta-programming.
  class NumPyInt16 {
#ifdef NPY_INT16
    public: typedef npy_int16 CType;
#endif
  };

  /// A meta-type for specifying a NumPy unsigned 32-bit int scalar for the
  /// purposes of generic meta-programming.
  class NumPyUInt32 {
#ifdef NPY_UINT32
    public: typedef npy_uint32 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 32-bit int scalar for the
  /// purposes of generic meta-programming.
  class NumPyInt32 {
#ifdef NPY_INT32
    public: typedef npy_int32 CType;
#endif
  };

  /// A meta-type for specifying a NumPy unsigned 64-bit int scalar for the
  /// purposes of generic meta-programming.
  class NumPyUInt64 {
#ifdef NPY_UINT64
    public: typedef npy_uint64 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 64-bit int scalar for the
  /// purposes of generic meta-programming.
  class NumPyInt64 {
#ifdef NPY_INT64
    public: typedef npy_int64 CType;
#endif
  };

  /// A meta-type for specifying a NumPy unsigned 128-bit int scalar
  /// for the purposes of generic meta-programming.
  class NumPyUInt128 {
#ifdef NPY_UINT128
    public: typedef npy_uint128 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 128-bit int scalar for the
  /// purposes of generic meta-programming.
  class NumPyInt128 {
#ifdef NPY_INT128
    public: typedef npy_int128 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 16-bit float scalar for the
  /// purposes of generic meta-programming.
  class NumPyFloat16 {
#ifdef NPY_FLOAT16
    public: typedef npy_int16 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 32-bit complex scalar for the
  /// purposes of generic meta-programming.
  class NumPyComplex32 {
#ifdef NPY_COMPLEX32
    public: typedef npy_complex32 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 32-bit float scalar for the
  /// purposes of generic meta-programming.
  class NumPyFloat32 {
#ifdef NPY_FLOAT32
    public: typedef npy_float32 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 64-bit complex scalar for the
  /// purposes of generic meta-programming.
  class NumPyComplex64 {
#ifdef NPY_COMPLEX64
    public: typedef npy_complex64 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 64-bit float scalar for the
  /// purposes of generic meta-programming.
  class NumPyFloat64 {
#ifdef NPY_FLOAT64
    public: typedef npy_float64 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 128-bit complex scalar for the
  /// purposes of generic meta-programming.
  class NumPyComplex128 {
#ifdef NPY_COMPLEX128
    public: typedef npy_complex128 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 96-bit float scalar for the
  /// purposes of generic meta-programming.
  class NumPyFloat96 {
#ifdef NPY_FLOAT96
    public: typedef npy_float96 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 192-bit complex scalar for the
  /// purposes of generic meta-programming.
  class NumPyComplex192 {
#ifdef NPY_COMPLEX192
    public: typedef npy_complex192 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 128-bit float scalar for the
  /// purposes of generic meta-programming.
  class NumPyFloat128 {
#ifdef NPY_FLOAT128
    public: typedef npy_float128 CType;
#endif
  };

  /// A meta-type for specifying a NumPy 256-bit complex scalar for the
  /// purposes of generic meta-programming.
  class NumPyComplex256 {
#ifdef NPY_COMPLEX256
    public: typedef npy_complex256 CType;
#endif
  };

  /// A very general templated class for encapsulating type
  /// information about NumPy arrays with specific element types. This
  /// class can be used for simple type checking of the elements in a
  /// NumPy array buffer. The template argument CPP is the type of the
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
  /// \tparam CPP The type of the CPP object.
  template<class CPP> struct NumPyType {

    /// The NumPy integer type code corresponding to objects of type
    /// CPP. By default, this is NPY_OBJECT. This is overridden by
    /// template specialization for each primitive numeric type in
    /// C++.
    static const int numpy_code = NPY_OBJECT;

    /// The name of the CPP type.
    static std::string cpp_typename(){ return "PyObject *"; }

    /// The Python representation of the NumPy dtype.
    static std::string numpy_dtype_str(){ return "np.object_"; }

    /// The type of the elements stored in the NumPy array's
    /// buffer. It's either a primitive or an array of PyObject*.
    typedef PyObject* ElemType;

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
    static bool isCompatible(const PyObject *obj) {
      return PyArray_Check(obj) && PyArray_TYPE(obj) == NPY_OBJECT;
    }

    /// Returns True if the NumPy arrays containing elements
    /// compatible with type CPP are stored as PyObjects.
    static bool isObjectArray() { return false; } \
  };

#define CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(ctype, pystruct) \
  template <class DefaultBufferAction> \
  struct DefaultToPythonStructure<ctype, DefaultBufferAction, 1> { \
    typedef pystruct Structure; \
  };

#define CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(Type, NumPyCode, NumPyTypeObject, NumPyDtypeString)	\
  /** Encapsulates type information for NumPy arrays of a specific NumPy buffer element type and corresponding C type. */ \
  template<> struct NumPyType<Type>			\
  {							\
    /** The NumPy type code. */                         \
    static const int   numpy_code = NumPyCode;			\
    /** The name of the corresponding C type as a string, which is useful for error reporting. */  \
    static std::string cpp_typename(){ return #Type;}		\
    /** The type of each element stored in the NumPy array buffer. */  \
    typedef Type ElemType;			\
    /** Converts a NumPy array element to a CPP array element. For simple primitive types (not PyObject*, e.g. int, float), these types are the same. */  \
    static bool isObjectArray() { return false; } \
    static std::string numpy_dtype_str(){ return NumPyDtypeString; } \
    static bool isCompatible(const PyObject *obj) { \
      return PyArray_Check(obj) && PyArray_TYPE(obj) == NumPyCode; \
    } \
  }

#define CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(npy_scalar_type_object, npy_scalar_pyobject_type, npy_scalar_ctype, npy_structure) \
  template <class CPPType> \
  struct ConvertToPython<CPPType, npy_structure> { \
    static PyObject* convert(const CPPType &val) { \
      npy_scalar_pyobject_type *obj = (npy_scalar_pyobject_type*)npy_scalar_type_object.tp_alloc(&npy_scalar_type_object, 0); \
      obj->obval = (npy_scalar_ctype)val; \
      return (PyObject*)obj; \
    } \
  }; \
  template <class CPPType> \
  struct ConvertToCPP<CPPType, npy_scalar_pyobject_type*, Copy<> > \
    : public ConvertToCPPBase<CPPType, Copy<> > { \
    ConvertToCPP() {} \
    void convert(PyObject *src, CPPType &dst) const { \
      npy_scalar_pyobject_type *obj((npy_scalar_pyobject_type *)src); \
      dst = (CPPType)obj->obval; \
      /**DefaultToCPPConvertActionUnresolved<npy_scalar_pyobject_type*, npy_scalar_ctype> f;**/ \
    } \
    virtual bool isImplemented() const { return true; } \
  };


#define CONVERTXY_DEFINE_NUMPY_COMPLEX_SCALAR_CONVERTER(npy_scalar_type_object, npy_scalar_pyobject_type, npy_scalar_ctype, npy_scalar_base_ctype, npy_structure) \
  template <class CPPType> \
  struct ConvertToPython<std::complex<CPPType>, npy_structure> { \
    static PyObject* convert(const std::complex<CPPType> &val) { \
      npy_scalar_pyobject_type *obj = (npy_scalar_pyobject_type*)npy_scalar_type_object.tp_alloc(&npy_scalar_type_object, 0); \
      (obj->obval).real = (npy_scalar_base_ctype)val.real(); \
      (obj->obval).imag = (npy_scalar_base_ctype)val.imag(); \
      return (PyObject*)obj; \
    } \
  }; \
  template <class CPPType> \
  struct ConvertToCPP<std::complex<CPPType>, npy_scalar_pyobject_type*, Copy<> > \
    : public ConvertToCPPBase<std::complex<CPPType>, Copy<> > { \
    ConvertToCPP() {} \
    void convert(PyObject *src, std::complex<CPPType> &dst) const { \
      npy_scalar_pyobject_type *obj((npy_scalar_pyobject_type*)src); \
      dst.real() = (CPPType)obj->obval.real; \
      dst.imag() = (CPPType)obj->obval.imag; \
    } \
    virtual bool isImplemented() const { return true; } \
  };

#define CONVERTXY_DEFINE_NUMPY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(npy_scalar_type_object, npy_noncomplex_scalar_pyobject_type, npy_complex_scalar_pyobject_type, npy_scalar_ctype, npy_scalar_base_ctype, npy_structure) \
  template <class CPPType> \
  struct ConvertToCPP<std::complex<CPPType>, npy_noncomplex_scalar_pyobject_type*, Copy<> > \
    : public ConvertToCPPBase<std::complex<CPPType>, Copy<> > { \
    ConvertToCPP() {} \
    void convert(PyObject *src, std::complex<CPPType> &dst) const { \
      npy_noncomplex_scalar_pyobject_type *obj((npy_noncomplex_scalar_pyobject_type*)src); \
      dst.real() = (CPPType)obj->obval; \
      dst.imag() = 0.0; \
    } \
    virtual bool isImplemented() const { return true; } \
  };

#define CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(type) \
    template <class DefaultBufferAction> \
      struct DefaultToCPPConvertAction<type, DefaultBufferAction, 1 > { typedef Copy<> Action; }; \
    template <> \
      struct Primitive<type, 1> { static const bool primitive = true; };

#define CONVERTXY_DEFINE_DEFAULT_NPY_COMPLEX_TOCPP_CONVERT_ACTION(type) \
    template <class DefaultBufferAction> \
      struct DefaultToCPPConvertAction<std::complex<type>, DefaultBufferAction, 1 > { typedef Copy<> Action; }; \
    template <> \
      struct Primitive<std::complex<type>, 1> { static const bool primitive = true; };


#define CONVERTXY_DEFINE_DEFAULT_NPY_COMPLEX_TOPYTHON_CONVERT_ACTION(ctype, pystruct) \
  template <class DefaultBufferAction> \
  struct DefaultToPythonStructure<std::complex<ctype>, DefaultBufferAction, 1> { \
    typedef pystruct Structure; \
  };

  //#define DOXYGEN_EXPAND_MACRO
  //#if defined(DOXYGEN_EXPAND_MACRO)
#ifdef NPY_BOOL
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_bool  , NPY_BOOL, PyBoolArrType_Type, "np.bool");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyBoolArrType_Type, PyBoolScalarObject, npy_bool, NumPyBool);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_bool, NumPyBool);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_bool);
#endif
#ifdef NPY_UINT8
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_uint8 , NPY_UINT8, PyUInt8ArrType_Type, "np.uint8");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyUInt8ArrType_Type, PyUInt8ScalarObject, npy_uint8, NumPyUInt8);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_uint8, NumPyUInt8);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_uint8);
#endif
#ifdef NPY_INT8
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_int8  , NPY_INT8, PyInt8ArrType_Type, "np.int8");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyInt8ArrType_Type, PyInt8ScalarObject, npy_int8, NumPyInt8);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_int8, NumPyInt8);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_int8);
#endif
#ifdef NPY_UINT16
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_uint16, NPY_UINT16, PyUInt16ArrType_Type, "np.uint16");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyUInt16ArrType_Type, PyUInt16ScalarObject, npy_uint16, NumPyUInt16);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_uint16, NumPyUInt16);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_uint16);
#endif
#ifdef NPY_INT16
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_int16 , NPY_INT16, PyInt16ArrType_Type, "np.int16");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyInt16ArrType_Type, PyInt16ScalarObject, npy_int16, NumPyInt16);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_int16, NumPyInt16);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_int16);
#endif
#ifdef NPY_UINT32
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_uint32, NPY_UINT32, PyUInt32ArrType_Type, "np.uint32");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyUInt32ArrType_Type, PyUInt32ScalarObject, npy_uint32, NumPyUInt32);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_uint32, NumPyUInt32);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_uint32);
#endif
#ifdef NPY_INT32
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_int32 , NPY_INT32, PyInt32ArrType_Type, "np.int32");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyInt32ArrType_Type, PyInt32ScalarObject, npy_int32, NumPyInt32);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_int32, NumPyInt32);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_int32);
#endif
#ifdef NPY_UINT64
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_uint64, NPY_UINT64, PyUInt64ArrType_Type, "np.uint64");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyUInt64ArrType_Type, PyUInt64ScalarObject, npy_uint64, NumPyUInt64);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_uint64, NumPyUInt64);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_uint64);
#endif
#ifdef NPY_INT64
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_int64 , NPY_INT64, PyInt64ArrType_Type, "np.int64");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyInt64ArrType_Type, PyInt64ScalarObject, npy_int64, NumPyInt64);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_int64, NumPyInt64);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_int64);
#ifdef NPY_FLOAT64
  CONVERTXY_DEFINE_NUMPY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(PyComplex128ArrType_Type, PyInt64ScalarObject, PyComplex128ScalarObject, npy_complex128, npy_int64, NumPyInt64);
#endif
#endif
#ifdef NPY_FLOAT32
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_float32, NPY_FLOAT32, PyFloat32ArrType_Type, "np.float32");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyFloat32ArrType_Type, PyFloat32ScalarObject, npy_float32, NumPyFloat32);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_float32, NumPyFloat32);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_float32);

  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_complex64, NPY_COMPLEX64, PyComplex64ArrType_Type, "np.cfloat64");
  CONVERTXY_DEFINE_NUMPY_COMPLEX_SCALAR_CONVERTER(PyComplex64ArrType_Type, PyComplex64ScalarObject, npy_complex64, npy_float32, NumPyFloat32);
  CONVERTXY_DEFINE_DEFAULT_NPY_COMPLEX_TOPYTHON_CONVERT_ACTION(npy_float32, NumPyComplex64);
  CONVERTXY_DEFINE_DEFAULT_NPY_COMPLEX_TOCPP_CONVERT_ACTION(npy_float32);
#endif
#ifdef NPY_FLOAT64
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_float64, NPY_FLOAT64, PyFloat64ArrType_Type, "np.float64");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyFloat64ArrType_Type, PyFloat64ScalarObject, npy_float64, NumPyFloat64);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_float64, NumPyFloat64);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_float64);

  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_complex128, NPY_COMPLEX128, PyComplex128ArrType_Type, "np.cfloat128");
  CONVERTXY_DEFINE_NUMPY_COMPLEX_SCALAR_CONVERTER(PyComplex128ArrType_Type, PyComplex128ScalarObject, npy_complex128, npy_float64, NumPyComplex128);
  CONVERTXY_DEFINE_NUMPY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(PyComplex128ArrType_Type, PyFloat64ScalarObject, PyComplex128ScalarObject, npy_complex128, npy_float64, NumPyFloat64);
  CONVERTXY_DEFINE_DEFAULT_NPY_COMPLEX_TOPYTHON_CONVERT_ACTION(npy_float64, NumPyComplex128);
  CONVERTXY_DEFINE_DEFAULT_NPY_COMPLEX_TOCPP_CONVERT_ACTION(npy_float64);
#endif
#ifdef NPY_FLOAT96
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_float96, NPY_FLOAT96, PyFloat96ArrType_Type, "np.float96");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyFloat96ArrType_Type, PyFloat96ScalarObject, npy_float96, NumPyFloat96);
  CONVERTXY_DEFINE_NUMPY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(PyComplex128ArrType_Type, PyFloat96ScalarObject, PyComplex192ScalarObject, npy_complex192, npy_float96, NumPyComplex192);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_float96, NumPyFloat96);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_float96);

  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_complex192, NPY_COMPLEX192, PyComplex192ArrType_Type, "np.cfloat192");
  CONVERTXY_DEFINE_NUMPY_COMPLEX_SCALAR_CONVERTER(PyComplex192ArrType_Type, PyComplex192ScalarObject, npy_complex192, npy_float96, NumPyComplex192);
  CONVERTXY_DEFINE_DEFAULT_NPY_COMPLEX_TOPYTHON_CONVERT_ACTION(npy_float96, NumPyComplex192);
  CONVERTXY_DEFINE_DEFAULT_NPY_COMPLEX_TOCPP_CONVERT_ACTION(npy_float96);
#endif
#ifdef NPY_FLOAT128
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_float128, NPY_FLOAT128, PyFloat128ArrType_Type, "np.float128");
  CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyFloat128ArrType_Type, PyFloat128ScalarObject, npy_float128, NumPyFloat128);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOPYTHON_CONVERT_ACTION(npy_float128, NumPyFloat128);
  CONVERTXY_DEFINE_DEFAULT_NPY_TOCPP_CONVERT_ACTION(npy_float128);

  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(npy_complex256, NPY_COMPLEX256, PyComplex256ArrType_Type, "np.cfloat256");
  CONVERTXY_DEFINE_NUMPY_COMPLEX_SCALAR_CONVERTER(PyComplex256ArrType_Type, PyComplex256ScalarObject, npy_complex256, npy_float128, NumPyComplex256);
  CONVERTXY_DEFINE_DEFAULT_NPY_COMPLEX_TOPYTHON_CONVERT_ACTION(npy_float128, NumPyComplex256);
  CONVERTXY_DEFINE_DEFAULT_NPY_COMPLEX_TOCPP_CONVERT_ACTION(npy_float128);
#endif
#ifdef NPY_OBJECT
  CONVERTXY_DEFINE_NUMPY_TYPE_CORRESPONDENCE(PyObject*, NPY_OBJECT, PyObjectArrType_Type, "np.object_");
  //CONVERTXY_DEFINE_NUMPY_SCALAR_CONVERTER(PyObjectArrType_Type, PyObjectScalarObject, PyObject*, NumPyObject);

  template <class CPPType, class Structure>
  struct ConvertToPython<CPPType, NumPyObject<Structure> > {
    static PyObject* convert(const CPPType &val) {
      PyObjectScalarObject *obj = PyObjectArrType_Type.tp_alloc(&PyObjectArrType_Type, 0);
      obj->obval = ConvertToPython<CPPType, Structure>::convert(val);
      return obj;
    }
  };
  template <class CPPType, class Action>
  struct ConvertToCPP<CPPType, PyObjectScalarObject*, Action>
    : public ConvertToCPPBase<CPPType, Action > {
    ConvertToCPP() {}
    void convert(PyObject *src, CPPType &dst) const {
      Py::Object _src(src);
      PyObjectScalarObject *obj = PyObjectArrType_Type.tp_alloc(&PyObjectArrType_Type, 0);
      ConvertToCPPBase<CPPType, Action> &converter(ToCPPDispatch<CPPType, Action>::getConverter(src));
      converter.convert(src, dst);
    }
    virtual bool isImplemented() const { return true; }
  };

#endif

#if defined(NPY_FLOAT96) && defined(NPY_FLOAT128)

  typedef TypeList<npy_float32,
		   TypeList<npy_float64,
		            TypeList<npy_float96,
			             TypeList<npy_float128, TypeListEnd> > > > NumPyFloatTypes;

  typedef TypeList<npy_complex64,
		   TypeList<npy_complex128,
		            TypeList<npy_complex192,
			             TypeList<npy_complex256, TypeListEnd> > > > NumPyComplexTypes;

#elif !defined(NPY_FLOAT96) && defined(NPY_FLOAT128)

  typedef TypeList<npy_float32,
		   TypeList<npy_float64,
			    TypeList<npy_float128, TypeListEnd> > > NumPyFloatTypes;

  typedef TypeList<npy_complex64,
		   TypeList<npy_complex128,
		            TypeList<npy_complex256, TypeListEnd> > > NumPyComplexTypes;

#elif defined(NPY_FLOAT96) && !defined(NPY_FLOAT128)

  typedef TypeList<npy_float32,
		   TypeList<npy_float64,
			    TypeList<npy_float96, TypeListEnd> > > NumPyFloatTypes;

  typedef TypeList<npy_complex64,
		   TypeList<npy_complex128,
			    TypeList<npy_complex192, TypeListEnd> > > NumPyComplexTypes;

#elif !defined(NPY_FLOAT96) && !defined(NPY_FLOAT128)

  typedef TypeList<npy_float32,
		   TypeList<npy_float64, TypeListEnd> > NumPyFloatTypes;

  typedef TypeList<npy_complex32,
		   TypeList<npy_complex64, TypeListEnd> > NumPyComplexTypes;

#endif

  template <class A, class B>
  struct Splice {};

  template <>
  struct Splice<TypeListEnd, TypeListEnd> {
    typedef TypeListEnd List;
  };

  template <class A>
  struct Splice<A, TypeListEnd > {
    typedef A List;
  };

  template <class B>
  struct Splice<TypeListEnd, B> {
    typedef B List;
  };

  template <class AHead, class ATail, class BHead, class BTail>
  struct Splice<TypeList<AHead, ATail>, TypeList<BHead, BTail> > {
    typedef typename Splice<ATail, BTail>::List Tails;
    typedef TypeList<AHead, TypeList<BHead, Tails> > List;
  };

  typedef Splice<NumPyFloatTypes, NumPyComplexTypes>::List NumPyFloatComplexTypes;

}

#endif
