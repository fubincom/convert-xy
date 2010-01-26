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

  /// A class for allocating buffers of a particular element type with
  /// <code>malloc</code>.
  ///
  /// \tparam ElemType The type of the elements in buffers to allocate
  /// or free with this allocator.
  template <typename ElemType>
  struct MallocAllocator {

    /// Allocates a buffer of <code>ElemType</code> of a particular
    /// size.
    ///
    /// @param size    The number of elements to allocate.
    ///
    /// @return        The allocated buffer (0 if error).
    static ElemType *allocate(size_t size) {
      return (ElemType*)calloc(size, sizeof(ElemType));
    }

    /// Frees a buffer allocated with this class.
    ///
    /// @param ptr     The buffer to free.
    static void free(ElemType *ptr) {
      free(ptr);
    }
  };

  /// A class for allocating buffers of a particular element type with
  /// the C++ <code>new</code> allocator.
  ///
  /// \tparam ElemType The type of the elements in buffers to allocate
  /// or free with this allocator.
  template <typename ElemType>
  struct CPPAllocator {
    
    /// Allocates a buffer of <code>ElemType</code> of a particular
    /// size.
    ///
    /// @param size    The number of elements to allocate.
    ///
    /// @return        The allocated buffer.
    ///
    /// @exception std::alloc   Thrown if an allocation error occurs.
    static ElemType *allocate(size_t size) {
      return new ElemType[size];
    }
    
    /// Frees a buffer allocated with this class.
    ///
    /// @param ptr     The buffer to free.
    static void free(ElemType *ptr) {
      delete []ptr;
    }
  };


  /// A templated class for allocating multidimensional
  /// C-arrays. The template arguments include the number of dimensions,
  /// element type, and one-dimensional allocator.
  ///
  /// For example, to allocate a 3-dimensional 10x20x30 array of
  /// floats,
  /// \code
  ///   int sz[3] {10, 20, 30};
  ///   int ***array = CArray<int, 3>::allocate(&sz);
  /// \endcode
  ///
  /// During allocation, first space is allocated to hold the child axes. It
  /// then instantiates itself recursively, incrementing the
  /// <code>Start</code> template parameter after allocating the
  /// current axis, which instantiates an allocator for allocating
  /// child axes.
  ///
  /// \tparam Dims      The number of dimensions to allocate for the buffers.
  /// \tparam ElemType  The type of the elements stored in multi-dimensional
  ///                   arrays stored.
  /// \tparam Allocator The allocator.
  ///
  /// @exception std::alloc   Thrown if an allocation error occurs.
  template <int Dims, typename ElemType,
	    template <class Q> class Allocator = DEFAULT_PYCPP_ALLOCATOR,
	    int Start = 0>
  struct CArray {

    /// The allocator class for elements of type <code>ChildType</code>.
    typedef CArray<Dims-1, ElemType, Allocator, Start+1> ChildClass;

    /// Higher-dimensional C-arrays are stored as arrays of
    /// arrays. The <code>ChildType</code> is the type of elements at
    /// dimension <code>Dim</code>.
    ///
    /// For one-dimensional arrays, <code>ChildType<code> is
    /// ElemType. For k-dimensional arrays, <code>ChildType</code> is
    /// (k-1) dimensional <code>ElemType</code> pointer.
    typedef typename ChildClass::MyType ChildType;

    /// A pointer to the elements stored at dimension <code>Dim</code> of
    /// an array allocated by this allocator.
    typedef ChildType* MyType;

    /// Allocates a <code>Dim</code>-dimensional array of type
    /// <code>ElemType</code>.
    ///
    /// The size of the array to allocate.
    static ChildType *allocate(int *sz) {
      // Grab the dimension size starting at Start.
      const int msz = sz[Start];

      // Allocate a buffer of ChildType's, returning a ChildType*.
      ChildType *me = Allocator<ChildType>::allocate(msz);

      // For each element of this multi-dimensional buffer, allocate
      // the child axis.
      for (int i = 0; i < msz; i++) {
	me[i] = ChildClass::allocate(sz);
      }

      // Return the final array.
      return me;
    }
  };

  /// A template class for allocating one-dimensional C-arrays. This
  /// is the base case for the <code>CArray</code> recursive template
  /// for allocating arbitrary multi-dimensional C-arrays.
  ///
  /// For example, to allocate a one-dimesional character array
  /// \code
  ///   int sz[] = {10};
  ///   CArray<1, char>::allocate(10);
  /// \endcode
  ///
  /// @param sz   The size of the one-dimensional array to allocate.
  ///
  /// @return     The allocated buffer.
  ///
  /// @exception std::alloc   Thrown if an allocation error occurs.
  template <typename ElemType,
	    template <class Q> class Allocator, int Start>
  struct CArray<1, ElemType, Allocator, Start> {

    typedef ElemType* MyType;

    static ElemType *allocate(int *sz) {
      ElemType *me = Allocator<ElemType>::allocate(sz[Start]);
      return me;
    }
  };

  /// A template specialization in case the base case array allocator
  /// isn't properly caught.
  template <typename ElemType,
	    template <class Q> class Allocator>
  class CArray<0, ElemType, Allocator> {
  private:
    /// A meta-compiler error message for when the user tries to allocate
    /// a zero dimensional array.
    class ZeroDimensionalArraysAreNotAllowed {};     

  public:
    /// This should cause a compiler error.
    static ElemType *allocate(ZeroDimensionalArraysAreNotAllowed &a);
  };

  /// A template specialization in case the base case array allocator
  /// isn't properly caught.
  template <typename ElemType,
	    template <class Q> class Allocator>
  class CArray<-1, ElemType, Allocator> {

  private:
    /// A meta-compiler error message for when the user tries to allocate
    /// a zero dimensional array.
    class NegativeDimensionalArraysAreNotAllowed {};     

  public:
    /// This should cause a compiler error.
    static ElemType *allocate(NegativeDimensionalArraysAreNotAllowed &a);
  };

  /// A template specialization in case the base case array allocator
  /// isn't properly caught.
  template <typename ElemType,
	    template <class Q> class Allocator>
  class CArray<-2, ElemType, Allocator> {

  private:
    /// A meta-compiler error message for when the user tries to allocate
    /// a zero dimensional array.
    class ZeroDimensionalArraysAreNotAllowed {};     

  public:
    /// This should cause a compiler error.
    static ElemType *allocate(ZeroDimensionalArraysAreNotAllowed &a);
  };
}
#endif
