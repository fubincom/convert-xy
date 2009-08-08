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

#ifndef SCTL_PYC_HPP
#define SCTL_PYC_HPP

/// \file   PyC.hpp
///
/// \brief  This header contains conversion functions and classes for
/// converting between NumPy arrays and multi-dimensional C arrays.
///
/// It also allows multidimensional arrays of any type to be allocated
/// and freed with a succinct syntax.
///
/// \author Damian Eads

#ifndef DEFAULT_PYCPP_ALLOCATOR
#define DEFAULT_PYCPP_ALLOCATOR MallocAllocator
#endif

#include <malloc.h>

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace PyCPP {

  template <typename ElemType>
  struct MallocAllocator {

    static ElemType *allocate(size_t size) {
      return (ElemType*)calloc(size, sizeof(ElemType));
    }

    static void free(ElemType *ptr) {
      free(ptr);
    }
  };

  template <typename ElemType>
  struct CPPAllocator {
    
    static ElemType *allocate(size_t size) {
      return new ElemType[size];
    }
    
    static void free(ElemType *ptr) {
      delete []ptr;
    }
  };


  template <int Dims, typename ElemType,
	    template <class Q> class Allocator = DEFAULT_PYCPP_ALLOCATOR,
	    int Start = 0>
  struct CArray {

    static typedef CArray<Dims-1, ElemType, Allocator, Start+1> ChildClass;

    static typedef typename ChildClass::MyType ChildType;

    static typedef ChildType* MyType;

    static ChildType *allocate(int *sz) {
      const int msz = sz[Start];
      ChildType *me = Allocator<ChildType>::allocate(msz);
      for (int i = 0; i < msz; i++) {
	me[i] = ChildClass::allocate(sz);
      }
      return me;
    }
  };

  template <typename ElemType,
	    template <class Q> class Allocator, int Start>
  struct CArray<1, ElemType, Allocator, Start> {

    static typedef ElemType* MyType;

    static ElemType *allocate(int *sz) {
      ElemType *me = Allocator<ElemType>::allocate(sz[Start]);
      return me;
    }
  };

  template <typename ElemType,
	    template <class Q> class Allocator>
  struct CArray<0, ElemType, Allocator> {};

  template <typename ElemType,
	    template <class Q> class Allocator>
  struct CArray<-1, ElemType, Allocator> {};

  template <typename ElemType,
	    template <class Q> class Allocator>
  struct CArray<-2, ElemType, Allocator> {};
}
#endif
