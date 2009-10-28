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
#include <cvd/image.h>

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace PyCPP {

  ///
  /// The classes of this namespace are used for defining the structure
  /// of a Python container.
  ///
  namespace Structure {

    /// Defines the structure of a Python list.
    ///
    /// For example,
    /// \code
    /// using namespace Structure;
    /// typedef List<Int> ListOfInts;
    /// \endcode
    /// defines a list of ints.
    ///
    /// \tparam T The structure of the elements of the list.
    template <class T>
    class List {};

    /// Defines the structure of a Python tuple.
    ///
    /// For example,
    /// \code
    /// using namespace Structure;
    /// typedef Tuple<Int> TupleOfInts;
    /// \endcode
    /// defines a tuple of ints.
    ///
    /// \tparam T The structure of the elements of the list.
    template <class T>
    class Tuple {};

    /// Defines the structure of a NumPy array.
    ///
    /// For example,
    /// \code
    /// typedef Array<Float>
    /// \endcode
    ///
    /// \tparam T The structure of the elements of the float.
    template <class T>
    class Array {
      typedef PyArrayObject PyObjectType;
    };

    /// Defines the structure of a dictionary.
    template <class K, class V>
    class Dict {
      typedef PyDictObject PyObjectType;
    };

    /// A terminal (no template arguments) structure type for python ints.
    class Integer {
      typedef PyIntObject PyObjectType;
    };

    /// A terminal (no template arguments) structure type for python longs.
    class Long {
      typedef PyLongObject PyObjectType;
    };

    /// A terminal (no template arguments) structure type for python floats.
    class Float {
      typedef PyFloatObject PyObjectType;
    };

    /// A terminal (no template arguments) structure type for python strings.
    class String {
      typedef PyStringObject PyObjectType;
    };
    
    /// A terminal (no template arguments) structure type for the python none object.
    class None {
      typedef PyObject PyObjectType;
    };

    /// A utility class for defining the default structure of a Python compound
    /// container given a CPP container.
    ///
    /// For example,
    /// \code
    ///   typedef map<string, vector<int> > MyMap;
    /// \endcode
    /// defines a type MyMap, which is just a STL map mapping strings to
    /// integers.
    /// 
    /// By instantiating a Default class with the C++ type as the template,
    /// argument, the member typedef Structure defines the default structure
    /// of the python argument.
    /// \code
    ///   typedef Default<MyMap>::Structure CorrespondingPyType;
    /// \endcode
    template <class CPP>
    class Default {};

    /// Defines the default structure of a Python object for a STL map.
    template <>
    template <class K, class V, class C, class A>
    class Default<std::map<K, V, C, A> > {
      typedef typename Default<K>::Structure KeyType;
      typedef typename Default<V>::Structure ValueType;
      typedef Dict<KeyType, ValueType > Structure;

      typedef PyDictObject PyObjectType;
    };

    /// Defines the default structure of a Python object for a STL vector.
    template <>
    template <class E, class A>
    class Default<std::vector<E, A> > {
      typedef typename Default<E>::Structure ElemType;
      typedef List<ElemType> Structure;

      typedef PyListObject PyObjectType;
    };

    /// Defines the default structure of a Python object for a TooN vector.
    template <>
    template <class E, int N, class B>
    class Default<TooN::Vector<N, E, B> > {
      typedef typename Default<E>::Structure ElemType;
      typedef Array<ElemType> Structure;

      typedef PyArrayObject PyObjectType;
    };

    /// Defines the default structure of a Python object for a TooN matrix.
    template <>
    template <class E, int M, int N, class B>
    class Default<TooN::Matrix<M, N, E, B> > {
      typedef typename Default<E>::Structure ElemType;
      typedef Array<ElemType> Structure;

      typedef PyArrayObject PyObjectType;
    };

    /// Defines the default structure of a Python object for a CVD image.
    template <>
    template <class T>
    class Default<CVD::Image<T> > {
      typedef typename Default<T>::Structure ElemType;
      typedef Array<ElemType> Structure;
    };
  }

  ///
  /// A general templated abstract base class for converting between
  /// Python and C++ objects but only with the C++ type specified.
  ///
  /// \tparam CPPType   The type of the C++ object.
  /// \tparam Structure The structure of the C++ object.
  template <class CPPType>
  class ToCPPBase {
  protected:
    /// Constructs an abstract converter.
    ToCPPBase() {}

    /// Destructs an abstract converter.
    ~ToCPPBase() {}

  public:

    /// Converts a Python object to a C++ object (with a copy).
    ///
    /// @param obj    The Python object to convert.
    virtual CPPType toCPP(PyObject *obj) = 0;

    /// Converts a Python object to a C++ object (without a copy).
    ///
    /// @param obj    The Python object to convert.
    virtual void toCPP(PyObject *obj, CPPType &cpp) = 0;

    /// Converts a Python object to a C++ object. 
    ///
    /// @param cpp    The C++ object to convert.
    virtual PyObject* toPython(const CPPType &cpp) = 0;
  };

  template <class CPPType, class PyType>
  class ToCPPBaseWithPyType : public ToCPPBase<CPPType> {
  public:
    ToCPPBaseWithPyType() {}
  };

  /// Serves a purpose only at compile-time for identifying
  /// when a to-python converter function could not be found.
  template <class CPPType>
  class CannotConvertToPython;

  template <typename CPP, typename PyStructure = typename Structure::Default<CPP>::Structure >
  struct ToPython { CannotConvertToPython<CPP> no_converter_found; };

  /// Serves a purpose only at compile-time for identifying
  /// when a to-c++ converter function could not be found.
  template <class CPPType, class PyType>
  class CannotConvertToCPP;

  template <class CPPType, class PyType>
  class ToCPP { CannotConvertToCPP<CPPType, PyType> no_converter_found; };

  /**
  template <>
  template <class T>
  class ToCPP<std::vector<T>, PyListObject*> : public ToCPPBaseWithPyType<CPPType, PyList*> {
    ToCPP() : ToCPPBaseWithPyType<CPPType, PyListObject*> {}

    virtual void convert(PyObject *obj, std::vector <T> &cpp) {
      cpp.clear();
      const int n = PyList_Size(obj);
      for (int i = 0; i < n; i++) {
	PyObject *pitem = PyList_GetItem(obj, i);
	toCPP(pitem);
	dst.push_back(toCPP(item));
      }
    }
  };
  **/

  /// Stores all conversion functions capable of converting a Python
  /// type to a CPPType. The converters are keyed with the unique
  /// hash of the Python type object.
  ///
  /// \tparam CPPType The type of the C++ object.
  template <class CPPType>
  class ToCPPDispatch {

  public:
    /// Converts the Python object <code>obj</code> to a C++
    /// object of type <code>CPPType</code>. If an appropriate
    /// converter cannot be found, an exception is thrown.
    ///
    /// @param obj The Python object to convert.
    ///
    /// @return The converted C++ object.
    static CPPType toCPP(PyObject *obj) {
      return get(obj).toCPP(obj);
    }

    static void toCPP(PyObject *obj, CPPType &cpp) {
      return get(obj).toCPP(obj, cpp);
    }

    static void add(PyTypeObject *obj, ToCPPBase<CPPType> &converter) {
      converters()[getTypeHash(obj)] = converter;
    }

    static std::map<long, ToCPPBase<CPPType> > &converters() {
      std::map<long, ToCPPBase<CPPType> > *c = new std::map<long, ToCPPBase<CPPType> >();
      return *c;
    }

    static ToCPPBase<CPPType> &get(long code) {
      typename std::map<long, ToCPPBase<CPPType> >::iterator it(converters().find(code));
      if (it == converters().end()) {
        throw std::string("Can't dispatch converter function!");
      }
      return it->second;
    }

    template <class PyTypeObject>
    static long getTypeHash(PyTypeObject *typ) {
      return typ->hash(typ);
    }

    template <class PyTypeObject>
    static ToCPPBase<CPPType> &get(PyTypeObject *typ) {
      return get(getTypeHash(typ));
    }

    static ToCPPBase<CPPType> &get(PyObject *obj) {
      PyTypeObject *typ = obj->ob_type;
      return get(typ);
    }

  private:
    //static std::map<long, ToCPPBase<CPPType> > converters;
  };

  template <class CPPType>
  CPPType toCPP(PyObject *obj) {
    return ToCPPDispatch<CPPType>::convert(obj);
  }

  template <class CPPType>
  void toCPP(PyObject *obj, CPPType &cpp) {
    ToCPPDispatch<CPPType>::convert(obj, cpp);
  }
}

#endif
