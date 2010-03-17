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

#ifndef CONVERTXY_STL_HPP
#define CONVERTXY_STL_HPP

#include "converter.hpp"
#include "scalars.hpp"

#include <vector>
#include <map>

///
/// The ConvertXY namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace ConvertXY {

  using namespace std;

  ///////////////////////////////////////////////////////////////////////////////
  /////////////// STL map conversion
  ///////////////////////////////////////////////////////////////////////////////

  template <class KeyType, class ValueType, class Compare, class Allocator, class BA>
  struct DefaultToCPPConvertAction<std::map<KeyType, ValueType, Compare, Allocator>, BA > {
    typedef DefaultToCPPConvertAction<KeyType, BA > KeyClass;
    typedef DefaultToCPPConvertAction<ValueType, BA > ValueClass;
    typedef typename KeyClass::Action KeyAction;
    typedef typename ValueClass::Action ValueAction;
    typedef Copy<KeyAction, ValueAction> Action;
  };

  template <>
  template <class KeyType, class ValueType, class Compare, class Allocator, class KeyAction, class ValueAction>
  struct ConvertToCPP<map<KeyType, ValueType, Compare, Allocator>,
		      Py::MapBase<Py::Object>, Copy<KeyAction, ValueAction> >
    : public ConvertToCPPBase<map<KeyType, ValueType, Compare, Allocator>,
			      Copy<KeyAction, ValueAction> > {

    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}
    
    virtual void convert(PyObject *src,
			 map<KeyType, ValueType, Compare, Allocator> &dst) const {
      Py::Mapping mp(src);
      Py::List seq(mp.items());

      for (size_t i = 0; i < seq.size(); i++) {
	Py::Tuple item(seq[i]);
	Py::Object key(item[0]);
	Py::Object value(item[1]);
	ConvertToCPPBase<KeyType, KeyAction> &keyConverter(ToCPPDispatch<KeyType, KeyAction>::getConverter(key.ptr()));
	ConvertToCPPBase<ValueType, ValueAction> &valueConverter(ToCPPDispatch<ValueType, ValueAction>::getConverter(value.ptr()));


	if (Primitive<KeyType>::primitive && Primitive<ValueType>::primitive) {
	  std::pair<KeyType, ValueType> p(0, 0);
	  keyConverter.convert(key.ptr(), p.first);
	  valueConverter.convert(value.ptr(), p.second);
	  dst.insert(p);
	}
	else if (Primitive<KeyType>::primitive && !Primitive<ValueType>::primitive) {
	  vector <size_t> valueDims(valueConverter.getSize(value.ptr()));
	  void *valuePtr = valueConverter.getBufferPointer(value.ptr());
	  std::pair<KeyType, ValueType> p(0, ObjectFactory<ValueType>::create(valueDims, valuePtr));
	  keyConverter.convert(key.ptr(), p.first);
	  valueConverter.convert(value.ptr(), p.second);
	  dst.insert(p);
	}
	else if (!Primitive<KeyType>::primitive && Primitive<ValueType>::primitive) {
	  vector <size_t> keyDims(keyConverter.getSize(key.ptr()));
	  void *keyPtr = keyConverter.getBufferPointer(key.ptr());
	  std::pair<KeyType, ValueType> p(ObjectFactory<KeyType>::create(keyDims, keyPtr), 0);
	  keyConverter.convert(key.ptr(), p.first);
	  valueConverter.convert(value.ptr(), p.second);
	  dst.insert(p);
	}
	else {
	  vector <size_t> keyDims(keyConverter.getSize(key.ptr()));
	  void *keyPtr = keyConverter.getBufferPointer(key.ptr());
	  vector <size_t> valueDims(valueConverter.getSize(value.ptr()));
	  void *valuePtr = valueConverter.getBufferPointer(value.ptr());
	  std::pair<KeyType, ValueType> p(ObjectFactory<KeyType>::create(keyDims, keyPtr),
					  ObjectFactory<ValueType>::create(valueDims, valuePtr));
	  keyConverter.convert(key.ptr(), p.first);
	  valueConverter.convert(value.ptr(), p.second);
	  dst.insert(p);
	}
      }
    }

    virtual bool isImplemented() const { return true; }

    void *getBufferPointer(PyObject *obj) const {
      return 0;
    }

    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> sz;
      Py::Sequence seq(src);
      sz.push_back(seq.size());
      return sz;
    }
  };

  template <class KeyType, class ValueType, class Compare,
	    class Allocator, class DefaultBufferAction>
  struct DefaultToPythonStructure<std::map<KeyType, ValueType, Compare, Allocator>, DefaultBufferAction > {
    typedef DefaultToPythonStructure<KeyType, DefaultBufferAction> KeyClass;
    typedef DefaultToPythonStructure<ValueType, DefaultBufferAction> ValueClass;
    typedef typename KeyClass::Structure KeyStructure;
    typedef typename ValueClass::Structure ValueStructure;
    typedef PyDict<KeyStructure, ValueStructure> Structure;
  };


  template <class KeyType, class ValueType,
	    class Compare, class Allocator,
	    class KeyStructure, class ValueStructure>
  struct ConvertToPython<map<KeyType, ValueType, Compare, Allocator>, PyDict<KeyStructure, ValueStructure> > {
    static PyObject* convert(const map<KeyType, ValueType, Compare, Allocator> &src) {
      Py::Dict my;
      for (typename map<KeyType, ValueType, Compare, Allocator>::const_iterator it(src.begin());
	   it != src.end(); it++) {
	my.setItem(Py::Object(ConvertToPython<KeyType, KeyStructure>::convert(it->first), true),
		   Py::Object(ConvertToPython<ValueType, ValueStructure>::convert(it->second), true));
      }
      return Py::new_reference_to(my);
    }
  };

  ///////////////////////////////////////////////////////////////////////////////
  /////////////// STL vector conversion
  ///////////////////////////////////////////////////////////////////////////////

  template <class ElemType, class Allocator, class BA>
  struct DefaultToCPPConvertAction<std::vector<ElemType, Allocator>, BA > {
    typedef DefaultToCPPConvertAction<ElemType, BA > ElemClass;
    typedef typename ElemClass::Action ElemAction;
    typedef Copy<ElemAction> Action;
  };

  template <>
  template <class ElemType, class Allocator, class ElemAction>
  struct ConvertToCPP<vector<ElemType, Allocator>, Py::SeqBase<Py::Object>, Copy<ElemAction> >
    : public ConvertToCPPBase<vector<ElemType, Allocator>, Copy<ElemAction> > {

    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}
    
    virtual void convert(PyObject *src, vector<ElemType, Allocator> &dst) const {
      //cerr << "hi!" << endl;
      Py::Sequence seq(src);

      for (size_t i = 0; i < seq.size(); i++) {
	//cerr << ".";
	Py::Object obj(seq[i]);
	ConvertToCPPBase<ElemType, ElemAction> &converter(ToCPPDispatch<ElemType, ElemAction>::getConverter(obj.ptr()));
	if (Primitive<ElemType>::primitive) {
	  dst.push_back(Zero<ElemType>::zero());
	  converter.convert(seq[i].ptr(), dst[dst.size() - 1]);
	}
	else {
	  vector <size_t> elemDims(converter.getSize(obj.ptr()));
	  void *elemPtr = converter.getBufferPointer(obj.ptr());
	  dst.push_back(ObjectFactory<ElemType>::create(elemDims, elemPtr));
	  converter.convert(seq[i].ptr(), dst[dst.size() - 1]);
	}
      }
    }

    virtual bool isImplemented() const { return true; }

    void *getBufferPointer(PyObject *obj) const {
      return 0;
    }

    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> sz;
      Py::Sequence seq(src);
      sz.push_back(seq.size());
      return sz;
    }
  };

  template <>
  template <class ElemType, class Allocator, class ElemAction>
  struct ConvertToCPP<vector<ElemType, Allocator>, PyIteratorProtocol, Copy<ElemAction> >
    : public ConvertToCPPBase<vector<ElemType, Allocator>, Copy<ElemAction> > {

    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}
    
    virtual void convert(PyObject *src, vector<ElemType, Allocator> &dst) const {
      //cerr << "hello!" << endl;
      PyObject *iterator = PyObject_GetIter(src);
      Py::Object pit(iterator, true); // so the iterator gets deleted when we're done

      PyObject *item(0);

      if (iterator == 0) {
	throw Py::TypeError("could not grab iterator!");
      }
      
      while ((item = PyIter_Next(iterator))) {
	Py::Object pitem(item, true);
	ConvertToCPPBase<ElemType, ElemAction>
	  &converter(ToCPPDispatch<ElemType, ElemAction>::getConverter(item));
	if (Primitive<ElemType>::primitive) {
	  dst.push_back(Zero<ElemType>::zero());
	  converter.convert(item, dst[dst.size() - 1]);
	}
	else {
	  vector <size_t> elemDims(converter.getSize(item));
	  void *elemPtr = converter.getBufferPointer(item);
	  dst.push_back(ObjectFactory<ElemType>::create(elemDims, elemPtr));
	  converter.convert(item, dst[dst.size() - 1]);
	}
      }
    }

    virtual bool isImplemented() const { return true; }

    void *getBufferPointer(PyObject *obj) const {
      return 0;
    }

    virtual vector <size_t> getSize(PyObject *src) const {
      vector <size_t> sz;
      return sz;
    }
  };

  template <class ElemType, class DefaultBufferAction>
  struct DefaultToPythonStructure<std::vector<ElemType>, DefaultBufferAction > {
    typedef DefaultToPythonStructure<ElemType, DefaultBufferAction> ElemClass;
    typedef typename ElemClass::Structure ElemStructure;
    typedef PyList<ElemStructure> Structure;
  };

  template <class ElemType, class ElemStructure>
  struct ConvertToPython<vector<ElemType>, PyList<ElemStructure> > {
    static PyObject* convert(const vector<ElemType> &src) {
      try {
	Py::List list(src.size());
	for (size_t i = 0; i < src.size(); i++) {
	  list[i] = Py::Object(ConvertToPython<ElemType, ElemStructure>::convert(src[i]), true);
	}
	return Py::new_reference_to(list);
      }
      catch (const Py::Exception &) {
	return 0;
      }
    }
  };

  template <class ElemType, class ElemStructure>
  struct ConvertToPython<vector<ElemType>, PyTuple<ElemStructure> > {
    static PyObject* convert(const vector<ElemType> &src) {
      try {
	Py::Tuple list(src.size());
	for (size_t i = 0; i < src.size(); i++) {
	  list[i] = Py::Object(ConvertToPython<ElemType, ElemStructure>::convert(src[i]), true);
	}
	return Py::new_reference_to(list);
      }
      catch (const Py::Exception &) {
	return 0;
      }
    }
  };

  template <class ElemType, class ElemStructure>
  struct ConvertToPython<vector<ElemType>, PySet<ElemStructure> > {
    static PyObject* convert(const vector<ElemType> &src) {
      try {
	PyObject *set;
	Py::List list(src.size());
	for (size_t i = 0; i < src.size(); i++) {
	  list[i] = Py::Object(ConvertToPython<ElemType, ElemStructure>::convert(src[i]), true);
	}
	set = PySet_New(list.ptr());
	if (set) {
	  return Py::new_reference_to(set);
	}
	else {
	  return 0;
	}
      }
      catch (const Py::Exception &) {
	return 0;
      }
    }
  };
}

#ifdef CONVERTXY_CVD_HPP
#include "CVDSTL.hpp"
#endif

#endif
