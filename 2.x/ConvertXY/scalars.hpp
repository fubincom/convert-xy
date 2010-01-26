#ifndef CONVERTXY_SCALARS_HPP
#define CONVERTXY_SCALARS_HPP

/// \file   scalars.hpp
///
/// \brief This header contains converter classes for converting
/// between Python and C scalars, generically.
///
/// \author Damian Eads

#include "converter.hpp"
#include <complex>

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace ConvertXY {


  template <class T, int ScalarCollectionLayer = 1>
  struct Primitive {
    typedef Primitive<T, ScalarCollectionLayer - 1> Predecessor;
    static const bool primitive = Predecessor::primitive;
  };

  template <class T>
  struct Primitive<T, 0> {
    static const bool primitive = false;
  };

#define CONVERT_XY_PRIM_TO_CPP_DEFAULTS(type) \
 template <class DefaultBufferAction> \
   struct DefaultToCPPConvertAction<type, DefaultBufferAction, 0 > { typedef Copy<> Action; }; \
 template <> \
   struct Primitive<type, 0> { static const bool primitive = true; };

  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(unsigned char);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(char);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(unsigned int);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(int);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(unsigned short);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(short);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(unsigned long);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(long);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(float);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(double);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(std::complex<float>);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(std::complex<double>);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(std::complex<long double>);
  CONVERT_XY_PRIM_TO_CPP_DEFAULTS(long double);

#define CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(ctype, pystruct)	\
  template <class DefaultBufferAction> \
  struct DefaultToPythonStructure<ctype, DefaultBufferAction, 0> { \
    typedef pystruct Structure; \
  };

  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(unsigned char,       PyInt);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(unsigned int,        PyInt);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(unsigned short,      PyInt);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(unsigned long,       PyLong);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(unsigned long long,  PyLong);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(char,                PyInt);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(int,                 PyInt);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(short,               PyInt);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(long,                PyLong);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(long long,           PyLong);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(float,               PyFloat);
  CONVERT_XY_PRIM_TO_PYTHON_DEFAULTS(double,              PyFloat);

#define CONVERT_XY_SCALAR_CONVERTER(ctype, pystruct, pycxxtype, casttype)	\
  template <> \
  struct ConvertToPython<ctype, pystruct> { \
    static PyObject* convert(const ctype &val) { \
      try { \
	pycxxtype ret((casttype)val);		  \
	return Py::new_reference_to(ret); \
      } \
      catch (const Py::Exception &) { \
	return 0; \
      } \
    } \
  }; \
  template <> \
  struct ConvertToCPP<ctype, pycxxtype, Copy<> > \
    : public ConvertToCPPBase<ctype, Copy<> > { \
    ConvertToCPP() {} \
    void convert(PyObject *src, ctype &dst) const { \
      pycxxtype el(src); \
      dst = (ctype)(casttype)el;			\
    } \
    virtual bool isImplemented() const { return true; } \
  };


#define CONVERT_XY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(ctype, pycxxtype, casttype)	\
  template <> \
  struct ConvertToCPP<std::complex<ctype>, pycxxtype, Copy<> >	\
    : public ConvertToCPPBase<std::complex<ctype>, Copy<> > {	\
    ConvertToCPP() {} \
    void convert(PyObject *src, std::complex<ctype> &dst) const {	\
      pycxxtype el(src); \
      dst = (ctype)(casttype)el;		\
    } \
    virtual bool isImplemented() const { return true; } \
  };

  CONVERT_XY_SCALAR_CONVERTER(unsigned char,       PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(unsigned int,        PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(unsigned short,      PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(unsigned long,       PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(unsigned long long,  PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(char,                PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(int,                 PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(short,               PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(long,                PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(long long,           PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(float,               PyInt, Py::Int, long);
  CONVERT_XY_SCALAR_CONVERTER(double,              PyInt, Py::Int, long);
  CONVERT_XY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(float, Py::Int, float);
  CONVERT_XY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(double,Py::Int, double);
  CONVERT_XY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(long double,Py::Int, long double);

  CONVERT_XY_SCALAR_CONVERTER(unsigned char,       PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(unsigned int,        PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(unsigned short,      PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(unsigned long,       PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(unsigned long long,  PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(char,                PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(int,                 PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(short,               PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(long,                PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(long long,           PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(float,               PyLong, Py::Long, long);
  CONVERT_XY_SCALAR_CONVERTER(double,              PyLong, Py::Long, long);

  CONVERT_XY_SCALAR_CONVERTER(float, PyFloat, Py::Float, double);
  CONVERT_XY_SCALAR_CONVERTER(double, PyFloat, Py::Float, double);
  CONVERT_XY_SCALAR_CONVERTER(long double, PyFloat, Py::Float, double);
  CONVERT_XY_SCALAR_CONVERTER(int, PyFloat, Py::Float, int);

  CONVERT_XY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(float, Py::Float, double);
  CONVERT_XY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(double, Py::Float, double);
  CONVERT_XY_NONCOMPLEX_TO_COMPLEX_SCALAR_CONVERTER(long double, Py::Float, double);

#define CONVERT_XY_FLOATPRIM_CONVERTER(ctype, pystruct, pycxxtype)	\
  template <> \
  struct ConvertToPython<ctype, pystruct> { \
    static PyObject* convert(const ctype &val) { \
      try { \
	pycxxtype ret((double)val);		  \
	return Py::new_reference_to(ret); \
      } \
      catch (const Py::Exception &) { \
	return 0; \
      } \
    } \
  }; \
  template <> \
  struct ConvertToCPP<ctype, pycxxtype, Copy<> > \
    : public ConvertToCPPBase<ctype, Copy<> > { \
    ConvertToCPP() {} \
    void convert(PyObject *src, ctype &dst) const { \
      pycxxtype el(src); \
      dst = (ctype)(double)el;			\
    } \
    virtual bool isImplemented() const { return true; } \
  };


  template <class DefaultBufferAction>
  struct DefaultToPythonStructure<string, DefaultBufferAction, 0> {
    typedef PyString Structure;
  };
    
  template <class DefaultBufferAction>
  struct DefaultToPythonStructure<char *, DefaultBufferAction, 0> {
    typedef PyString Structure;
  };

  template <class DefaultBufferAction>
  struct DefaultToCPPConvertAction<string, DefaultBufferAction, 0> {
    typedef Copy<> Action;
  };

  template <>
  struct ConvertToPython<string, PyString > {
    static PyObject* convert(const string &val) {
      try {
	Py::String ret(val);
	return Py::new_reference_to(ret);
      }
      catch (const Py::Exception &) {
	return 0;
      }
    }
  };

  template <>
  struct ConvertToPython<char *, PyString > {
    static PyObject* convert(const char *&val) {
      try {
	Py::String ret(val);
	return Py::new_reference_to(ret);
      }
      catch (const Py::Exception &) {
	return 0;
      }
    }
  };

  /////////////////////////////////////////////////////////////////////
  ///////////// To C++
  /////////////////////////////////////////////////////////////////////

  template <>
  struct ConvertToCPP<string, Py::String, Copy<> >
    : public ConvertToCPPBase<string, Copy<> > {

    ConvertToCPP() {}
    
    void convert(PyObject *src, string &dst) const {
      Py::String s(src);
      dst = s.as_std_string();
    }

    virtual bool isImplemented() const { return true; }

  };

#define CONVERT_XY_STRING_TO_INT_PRIM(T) \
  template <> \
  struct ConvertToCPP<T, Py::String, Copy<> >	    \
    : public ConvertToCPPBase<T, Copy<> > { \
    ConvertToCPP() {} \
    void convert(PyObject *src, T &dst) const { \
      Py::String _src(src); \
      istringstream istr(_src.as_std_string()); \
      istr >> dst; \
      if (istr.fail()) {\
         throw Py::TypeError("ConvertToCPP<Number, Py::String>: formatting error!"); \
      }\
    }\
    virtual bool isImplemented() const { return true; }\
  };

  CONVERT_XY_STRING_TO_INT_PRIM(float);
  CONVERT_XY_STRING_TO_INT_PRIM(double);
  CONVERT_XY_STRING_TO_INT_PRIM(char);
  CONVERT_XY_STRING_TO_INT_PRIM(short);
  CONVERT_XY_STRING_TO_INT_PRIM(int);
  CONVERT_XY_STRING_TO_INT_PRIM(long);
  CONVERT_XY_STRING_TO_INT_PRIM(long long);
  CONVERT_XY_STRING_TO_INT_PRIM(unsigned char);
  CONVERT_XY_STRING_TO_INT_PRIM(unsigned short);
  CONVERT_XY_STRING_TO_INT_PRIM(unsigned int);
  CONVERT_XY_STRING_TO_INT_PRIM(unsigned long);
  CONVERT_XY_STRING_TO_INT_PRIM(unsigned long long);

  struct TypeListEnd{};

  template<class C, class D> struct TypeList {
    typedef C type;
    typedef D next;
    typedef C Head;
    typedef D Tail;

  };

  typedef TypeList<double,
		   TypeList<float, TypeListEnd> > FloatTypes;

  typedef TypeList<double,
          TypeList<float,
	  TypeList<int,
	  TypeList<unsigned char,
	  TypeList<char,
	  TypeList<long,
          TypeList<long long,
	  TypeList<short,
	  TypeList<unsigned int,
	  TypeList<unsigned long,
	  TypeList<unsigned long long, TypeListEnd> > > > > > > > > > > NumberTypes;

  template <class T>
  struct SingletonList {
    typedef TypeList<T, TypeListEnd> List;
  };

  template <class WalkableTypeList>
  struct AdvanceIfPossible {};

  template <class A, class B>
  struct AdvanceIfPossible<TypeList<A, B> > {
    typedef B choice;
  };

  template <class A>
  struct AdvanceIfPossible<TypeList<A, TypeListEnd> > {
    typedef TypeList<A, TypeListEnd> choice;
  };

  template <class T>
  struct IsEnd {
    static const bool end = false;
  }; 

  template <>
  struct IsEnd<TypeListEnd> {
    static const bool end = true;
  };

}

#endif
