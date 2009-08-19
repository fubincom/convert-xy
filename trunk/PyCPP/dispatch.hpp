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

#ifndef SCTL_DISPATCH_HPP
#define SCTL_DISPATCH_HPP

/// \file   dispatch.hpp
///
/// \brief This header contains a polymorphic interface for calling
/// converter functions and specifying or statically infering
/// the structure of a python object based on a CPP object.
///
/// \author Damian Eads

#include <vector>
#include <map>
#include <string>
#include <TooN/TooN.h>

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace PyCPP {

  namespace PyTypes {

    template <class T>
    class PyList {};

    template <class T>
    class PyTuple {};

    template <class T>
    class PyArray {};
  
    template <class K, class V>
    class PyDict {};

    template <class CPP>
    class PyDefault {};

    template <>
    template <class T>
    class PyDefault<std::vector<T> > {
      typedef typename PyDefault<T>::PyStructure PyElementType;
      typedef PyList<PyElementType> PyStructure;
    };

    template <>
    template <class K, class V, class C, class A>
    class PyDefault<std::map<K, V, C, A> > {
      typedef typename PyDefault<K>::PyStructure PyKeyType;
      typedef typename PyDefault<V>::PyStructure PyValueType;
      typedef PyDict<PyKeyType, PyValueType > PyStructure;
      typedef PyDictObject PyType;
    };

    template <>
    template <class T, int N, class B>
    class PyDefault<TooN::Vector<N, T, B> > {
      typedef typename PyDefault<T>::PyStructure PyElementType;
      typedef PyArray<PyElementType> PyStructure;
    };
  }

  ///
  /// A general templated abstract base class for converting between
  /// Python and C++ objects but only with the C++ type specified.
  ///
  template <class CPPType, class PyStructure = PyTypes::PyDefault<CPPType> >
  class AbstractConverterBase {
  public:    
    AbstractConverterBase() {}
    ~AbstractConverterBase() {}

    virtual CPPType toCPP(PyObject *obj) = 0;
    virtual void toCPP(PyObject *obj, CPPType &cpp) = 0;
    virtual PyObject* toPython(const CPPType &cpp) = 0;

  };

  template <class CPPType, class PyStructure = PyTypes::PyDefault<CPPType> >
  class ConverterBase : public AbstractConverterBase<CPPType> {

    ConverterBase(PyTypeObject *typ) : AbstractConverterBase<CPPType>() {
      add(typ, this);
    }

  };

  template <class CPPType, class PyStructure = PyTypes::PyDefault<CPPType> >
  class Converter2 : public ConverterBase<PyObjectType, CPPType> {
    
    Converter2() : ConverterBase<PyObjectType, CPPType>() {}

    virtual CPPType toCPP(PyObject *obj) {
      return Converter<PyObjectType, CPPType>::convert(obj);
    }

    virtual void toCPP(PyObject *obj, CPPType &cpp) {
      Converter<PyObjectType, CPPType>::convert(obj, cpp);
    }

    virtual PyObject* toPython(const CPPType &cpp) {
      return (PyObject*)Converter<CPPType, PyObjectType*>::convert(cpp);
    }
  };


  template <class CPPType>
  class ConverterDispatch {

  public:

    static CPPType toCPP(PyObject *obj) {
      return get(obj).toCPP(obj);
    }

    static void toCPP(PyObject *obj, CPPType &cpp) {
      return get(obj).toCPP(obj, cpp);
    }

    static void add(PyTypeObject *obj, AbstractConverterBase<CPPType> &converter) {
      converters[getTypeHash(obj)] = converter;
    }

    static AbstractConverterBase<CPPType> &get(long code) {
      typename std::map<long, AbstractConverterBase<CPPType> >::iterator it(converters.find(code));
      if (it == converters.end()) {
        throw std::string("Can't dispatch converter function!");
      }
      return it->second;
    }

    template <class PyTypeObject>
    static long getTypeHash(PyTypeObject *typ) {
      return typ->hash(typ);
    }

    template <class PyTypeObject>
    static AbstractConverterBase<CPPType> &get(PyTypeObject *typ) {
      return get(getTypeHash(typ));
    }

    static AbstractConverterBase<CPPType> &get(PyObject *obj) {
      PyTypeObject *typ = obj->ob_type;
      return get(typ);
    }

  private:
    static std::map <long, AbstractConverterBase<CPPType> > converters;
  };

}

#endif
