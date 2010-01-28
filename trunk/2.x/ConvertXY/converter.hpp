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

/// \file   converter.hpp
///
/// \brief This header contains converter classes for converting
/// between Python and C scalars, generically.
///
/// \author Damian Eads

#ifndef CONVERTXY_CONVERTER_HPP
#define CONVERTXY_CONVERTER_HPP

#include <map>
#include <vector>
#include <string>

#include <CXX/Objects.hxx>

namespace ConvertXY {

  using namespace std;

  #define PYSEQUENCE_PROTOCOL -1
  #define PYMAPPING_PROTOCOL -2
  #define PYITERATOR_PROTOCOL -3

  template <class T>
  struct ObjectFactory {

    static T create(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      return T();
    }

    static T* create_ptr(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      return new T();
    }
  };

  template <class PyPointerType>
  class UnsupportedPythonType;

  template <class PyPointerType>
  struct PythonType {
    UnsupportedPythonType<PyPointerType> nothing_defined;
  };

  class NIL{};

  template <class ElemStructure>
  class PyList {};

  template <class ElemStructure>
  class PySet {};

  template <class ElemStructure>
  class PyTuple {};

  template <class KeyStructure, class ValueStructure>
  class PyDict {};

  template <class NumPyType, class BufferAction>
  class PyArray {};

  class PyString {};

  class PyLong {};
  class PyInt {};
  class PyFloat {};
  class PyIteratorProtocol {};

  /**template <class Head, class Tail>
  struct TypeCons {
    typedef Head head;
    typedef Tail tail;
  };

  typedef TypeCons<PyList_Type, NIL> PythonTypes;**/

  // Allocate: don't reuse any buffers from the source. Allocate a new buffer in the target.
  template <class X1 = NIL, class X2 = NIL, class X3 = NIL,
	    class X4 = NIL, class X5 = NIL, class X6 = NIL>
  struct Allocate {};

  // Reuse: reuse the buffer (ie copy the buffer pointer) from the source.
  template <class X1 = NIL, class X2 = NIL, class X3 = NIL,
	    class X4 = NIL, class X5 = NIL, class X6 = NIL>
  struct Reuse {};

  // Copy: don't reuse any buffers from the source. Use the target's
  // existing buffer and copy the data values from the source into it,
  // performing necessary type conversion if necessary.
  template <class X1 = NIL, class X2 = NIL, class X3 = NIL,
	    class X4 = NIL, class X5 = NIL, class X6 = NIL>
  struct Copy {};

  // Copy: don't reuse any buffers from the source. Allocate the target's
  // buffer and copy the data values from the source into it,
  // performing necessary type conversion if necessary.
  template <class X1 = NIL, class X2 = NIL, class X3 = NIL,
	    class X4 = NIL, class X5 = NIL, class X6 = NIL>
  struct AllocateCopy {};

  // Check that the size of the containers are the same.
  template <class X1 = NIL, class X2 = NIL, class X3 = NIL,
	    class X4 = NIL, class X5 = NIL, class X6 = NIL>
  struct CheckSize {};

  // Check that each key element from the Python mapping
  // exists in the C++ map
  struct CheckExists {};

  template <class CPPType>
  struct DefaultDefaultBufferAction {
    typedef Reuse<> DefaultBufferAction;
  };

  template <class CPPType, class DefaultBufferAction>
  struct DefaultToCPPConvertActionUnresolved;

  template <class CPPType,
	    class DefaultBufferAction, int ScalarCollectionLayer = 1>
  struct DefaultToCPPConvertAction {
    typedef DefaultToCPPConvertAction<CPPType, DefaultBufferAction, ScalarCollectionLayer - 1> Predecessor;
    typedef typename Predecessor::Action Action;
  };

  template <class CPPType,
	    class DefaultBufferAction>
  struct DefaultToCPPConvertAction<CPPType, DefaultBufferAction, 0> {
    DefaultToCPPConvertActionUnresolved<CPPType, DefaultBufferAction> cannot_resolve_convert_action;
  };

  template <class CPPType, class DefaultBufferAction>
  struct DefaultToPythonStructureUnresolved;

  template <class CPPType, class DefaultBufferAction, int ScalarCollectionLayer = 1>
  struct DefaultToPythonStructure {
    typedef DefaultToPythonStructure<CPPType, DefaultBufferAction, ScalarCollectionLayer - 1> Predecessor;
    typedef typename Predecessor::Structure Structure;
  };

  template <class CPPType, class DefaultBufferAction>
  struct DefaultToPythonStructure<CPPType, DefaultBufferAction, 0> {
    DefaultToPythonStructureUnresolved<CPPType, DefaultBufferAction> cannot_resolve_convert_structure;
  };

  template <class CPPType>
  class CannotReuse;

  template <class CPPType, class CopyAction>
  struct CannotConvertToPython;

  template <class CPPType, class Structure>
  struct ConvertToPython {
    CannotConvertToPython<CPPType, Structure> unsupported_type_or_structure;
  };

  template <class CPPType, class CopyAction>
  struct CannotConvertToCPP;

  template <class CPPType,
	    class CopyAction
	    = typename DefaultToCPPConvertAction<CPPType, Reuse<> >::Action >
  struct ConvertToCPPBase {
    ConvertToCPPBase() {}
    virtual ~ConvertToCPPBase() {}

    virtual void convert(PyObject *in, CPPType &out) const = 0;
    virtual bool isImplemented() const { return false; };
    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> v;
      v.push_back(1);
      return v;
    };
    virtual void *getBufferPointer(PyObject *src) const {
      return 0;
    }
  };

  template <class CPPType, class PyType,
	    class CopyAction
	    = typename DefaultToCPPConvertAction<CPPType, Reuse<> >::Action >
  struct ConvertToCPP : public ConvertToCPPBase<CPPType, CopyAction> {

    ConvertToCPP() {}
    
    void convert(PyObject *src, CPPType &dst) const {
      throw std::string("ConvertToCPP<General>::convert() Unsupported conversion between types!");
    }
  };


  template <class CPPType,
	    class CopyAction>
  struct ToCPPDispatch;

  template <class CPPType, class CopyAction, int r>
  struct DispatchPopulator {
    static void populate_dispatch_map(ToCPPDispatch<CPPType, CopyAction> &d) {
      DispatchPopulator<CPPType, CopyAction, r - 1>::populate_dispatch_map(d);
    }
  };

  template <class CPPType, class CopyAction>
  struct DispatchPopulator<CPPType, CopyAction, 0> {
    static void populate_dispatch_map() {
      typedef ToCPPDispatch<CPPType, CopyAction> D;
      D::add(PYSEQUENCE_PROTOCOL,
	  new ConvertToCPP<CPPType, Py::SeqBase<Py::Object>, CopyAction>());
      D::add(PYMAPPING_PROTOCOL,
	  new ConvertToCPP<CPPType, Py::MapBase<Py::Object>, CopyAction>());
      D::add(PYITERATOR_PROTOCOL,
	  new ConvertToCPP<CPPType, PyIteratorProtocol, CopyAction>());
      D::add(PyList_Type, new ConvertToCPP<CPPType, Py::List, CopyAction>());
      D::add(PyTuple_Type, new ConvertToCPP<CPPType, Py::Tuple, CopyAction>());
      D::add(PyDict_Type, new ConvertToCPP<CPPType, Py::Dict, CopyAction>());
      D::add(PyString_Type, new ConvertToCPP<CPPType, Py::String, CopyAction>());
      D::add(PySet_Type, new ConvertToCPP<CPPType, PySetObject*, CopyAction>());
      D::add(PyLong_Type, new ConvertToCPP<CPPType, Py::Long, CopyAction>());
      D::add(PyFloat_Type, new ConvertToCPP<CPPType, Py::Float, CopyAction>());
      D::add(PyInt_Type, new ConvertToCPP<CPPType, Py::Int, CopyAction>());
    }
  };

  template <class CPPType,
	    class CopyAction>
  struct ToCPPDispatch {

    typedef map<long long, ConvertToCPPBase<CPPType, CopyAction>* > DispatchMap;
    
    static DispatchMap &getMap() {
      static DispatchMap *m = 0;
      if (m == 0) {
	m = new DispatchMap();
	populate();
      }
      return *m;
    }

    static ConvertToCPPBase<CPPType, CopyAction> &getConverter(PyObject *src) {
      DispatchMap &dispatch_map(getMap());
      /**cerr << "(" << (long long)&PyFloat96ArrType_Type << "," << (long long)Py_TYPE(src) << ")";
	 cerr << "[" << (long long)&PyInt_Type << "," << (long long)Py_TYPE(src) << "]";**/

      typename DispatchMap::iterator it(dispatch_map.find((long long)Py_TYPE(src)));

      if (it == dispatch_map.end() || !it->second->isImplemented()) {
	typename DispatchMap::iterator seq_it(dispatch_map.find(PYSEQUENCE_PROTOCOL));
	typename DispatchMap::iterator map_it(dispatch_map.find(PYMAPPING_PROTOCOL));
	typename DispatchMap::iterator it_it(dispatch_map.find(PYITERATOR_PROTOCOL));
	PyObject *python_iter(0);
	if (seq_it != dispatch_map.end() && PySequence_Check(src)) {
	  return *(seq_it->second);
	}
	else if (map_it != dispatch_map.end() && PyMapping_Check(src)) {
	  return *(map_it->second);
	}
	else if (it_it != dispatch_map.end() && (python_iter = PyObject_GetIter(src))) {
	  Py::Object ppython_iter(python_iter, true);
	  return *(it_it->second);
	}
	else {
	  throw Py::TypeError("Cannot dispatch converter.");
	}
      }
      return *(it->second);
    }

    static long long getId(PyTypeObject &type) {
      return (long long)&type;
    }

    static void add(PyTypeObject &type, ConvertToCPPBase<CPPType, CopyAction> *ptr) {
      DispatchMap &m(getMap());
      long long id = getId(type);
      if (ptr->isImplemented()) {
	m.insert(std::pair<long long, ConvertToCPPBase<CPPType, CopyAction>* >(id, ptr));
      }
      else {
	delete ptr;
      }
    }


    static void add(long long id, ConvertToCPPBase<CPPType, CopyAction> *ptr) {
      DispatchMap &m(getMap());
      if (ptr->isImplemented()) {
	m.insert(std::pair<long long, ConvertToCPPBase<CPPType, CopyAction>* >(id, ptr));
      }
      else {
	delete ptr;
      }
    }

    static void populate() {
      DispatchPopulator<CPPType, CopyAction, 1>::populate_dispatch_map();
    }
  };

  template <class SrcType>
  void convert(SrcType &src, PyObject *&dest) {
    typedef typename DefaultToPythonStructure<SrcType, Copy<> >::Structure DefaultStructure;
    dest = ConvertToPython<SrcType, DefaultStructure>::convert(src);
  }

  template <class SrcType>
  PyObject *convert(const SrcType &src) {
    typedef typename DefaultToPythonStructure<SrcType, Copy<> >::Structure DefaultStructure;
    return ConvertToPython<SrcType, DefaultStructure>::convert(src);
  }

  template <class Structure, class SrcType>
  PyObject *convert_override(const SrcType &src) {
    return ConvertToPython<SrcType, Structure>::convert(src);
  }

  template <class DestType>
  void convert(PyObject *src, DestType &dest) {
    typedef typename DefaultToCPPConvertAction<DestType, Reuse<> >::Action ConvertAction;
    ConvertToCPPBase<DestType, ConvertAction> &converter(ToCPPDispatch<DestType, ConvertAction>::getConverter(src));
    converter.convert(src, dest);
  }

  template <class ConvertAction, class DestType>
  void convert_override(PyObject *src, DestType &dest) {
    ConvertToCPPBase<DestType, ConvertAction> &converter(ToCPPDispatch<DestType, ConvertAction>::getConverter(src));
    converter.convert(src, dest);
  }

  template <class T>
  struct IsNIL {
    static const bool nil = false;
  }; 

  template <>
  struct IsNIL<NIL> {
    static const bool nil = true;
  };

  template <class T, int ScalarCollectionLayer = 1>
  struct Primitive {
    typedef Primitive<T, ScalarCollectionLayer - 1> Predecessor;
    static const bool primitive = Predecessor::primitive;
  };

  template <class T>
  struct Primitive<T, 0> {
    static const bool primitive = false;
  };

  template <class T, bool IsPrimitive = Primitive<T>::primitive>
  struct Zero {
    static T zero() {
      return (T)0;
    }
  };

  template <bool IsPrimitive>
  struct Zero<std::string, IsPrimitive> {
    static std::string zero() {
      return "";
    }
  };

  template <class T>
  struct Zero<T, true> {
    static T zero() {
      return T();
    }
  };
}

#endif
