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

#ifndef SCTL_CONVERTER_H
#define SCTL_CONVERTER_H

/// \file   converter.hpp
///
/// \brief This header contains the main <code>Converter</code> class
/// template, the templated <code>convert</code> convert function, and
/// templated functions for retrieving dimension information from
/// Python container objects.
///
/// \author Damian Eads

#include <string>
#include <numpy/arrayobject.h>

///
/// The PyCPP namespace encapsulates templated functions and classes
/// for conversion of containers between C++ and Python.
///

namespace PyCPP {

  /// A very general templated class for allocating a 2D array buffer
  /// and pointing to it from two different data structures. Partial
  /// template specializations should provide two member functions.
  ///
  /// \code
  ///   BiAllocator<T1,T2,E,S>(const SizeSpecification &a)
  ///   Type1 &getFirst();
  ///   Type2 &getSecond();
  ///   const Type1 &getFirst() const;
  ///   const Type2 &getSecond() const;
  /// \endcode
  ///
  ///
  /// \tparam SizeSpecification  The type of the size specifier (e.g. int or pair<int, int>)
  /// \tparam Type1              The type to convert from.
  /// \tparam Type2              The type to convert to.
  template <typename Type1, typename Type2, typename Elem, typename SizeSpecification>
  struct BiAllocator {};

  /// A class just used for generating a compiler error when no
  /// converter can be found.
  template <class From, class To>
  struct NoConverterForTypes;

  /// A very general templated class for performing conversions
  /// between types. This is the most pervasive class of this template
  /// library. Partial template specializations of this class should
  /// provide two static member functions:
  ///
  /// \code
  ///   static void convert(const From &src, To &dst); // Form A
  ///   static To convert(const From &src);            // Form B
  /// \endcode
  ///
  /// The const modifier is optional for the <code>src</code> argument.
  /// It is usually is omitted when converting from a
  /// <code>PyObject*</code> since Python C library functions do
  /// not take arguments of type <code>const PyObject*</code> .
  ///
  /// Form A is generally preferable over B because it involves one
  /// less copy of an object of type <code>To</code>.
  ///
  /// If a compiler error results when instantiating a converter
  /// function, some parts of the conversion might not be supported.
  ///
  /// \tparam From    The type to convert from.
  /// \tparam To      The type to convert to.
  template <typename From, typename To>
  struct Converter {
    NoConverterForTypes<From, To> converter_not_found;
  };

  /// The identity conversion class for converting from a PyObject*
  /// to a PyObject*.
  template <>
  struct Converter<PyObject*, PyObject*> {

    /// The conversion simply involves returning the PyObject pointer
    /// passed.
    ///
    /// @param src    A pointer to the PyObject to convert.
    ///
    /// @return       The pointer passed.
    static PyObject* convert(PyObject *src) {
      return src;
    }


    /// The conversion simply involves copying the pointer
    /// passed as the source.
    ///
    /// @param src    A pointer to the PyObject to convert.
    /// @param dst    A pointer to the destination PyObject.
    ///
    /// @return       The pointer passed.
    static void convert(PyObject *src, PyObject *&dst) {
      dst = src;
    }
  };

  /// Converts a generic PyObject to an object of type T. At run-time,
  /// the function inspects the run-time type of the Python object and
  /// invokes the converter function for that type.
  ///
  /// This specific specialiation class needs a major overhaul, which
  /// is in progress (see dispatch.hpp).
  template <typename T>
  struct Converter<PyObject*, T> {

    static T convert(PyObject *src) {
      if (PyArray_Check(src)) {
	return Converter<PyArrayObject*, T>::convert((PyArrayObject*)src);
      }
      else {
	throw std::string("Cannot convert object!");
      }
    }

    static void convert(PyObject *src, T &dst) {
      if (PyArray_Check(src)) {
	Converter<PyArrayObject*, T>::convert(src, dst);
      }
      else {
	throw std::string("Cannot convert object!");
      }
    }
  };

  /// A templated function for invoking a Converter class's convert
  /// function. When the C++ template pattern matching facility
  /// recognizes the types of the arguments passed, arbitrary nested
  /// containers can be converted with a succinct <code>convert(x,
  /// y)</code> syntax.
  ///
  /// Here is a simple example of converting a map mapping strings
  /// to vectors of ints to a Python dictionary mapping Python
  /// strings to Python lists of ints.
  /// \code
  ///    void tryIt(const map<string, vector<int> > &ids) {
  ///       PyDictObject *pydict;
  ///       convert(ids, pydict);
  ///    }
  /// \endcode

  template <typename From, typename To>
  void convert(const From &from, To &to) {
    Converter<From, To>::convert(from, to);
  }

  /// A very general templated class for determining the length,
  /// number of rows, number of columns, number of dimensions and
  /// dimension size of Python and NumPy seqeunces.
  ///
  /// All Python sequence types are expected to implement
  /// any one or all of the following:
  /// \code
  ///   int len(PyXXXObject* src)
  ///   int ndim(PyXXXObject* src)
  ///   int dim(PyXXXObject* src, i)
  ///   int rows(PyXXXObject* src)
  ///   int cols(PyXXXObject* src)
  /// \endcode
  ///
  ///
  /// For example, to determine the number of rows and columns of a NumPy
  /// array <code>myArray</code>, do:
  /// \code
  ///    r = rows(myArray)
  ///    c = cols(myArray)
  /// \endcode
  ///
  /// If a compiler error occurs during template instantiate,
  /// the operation may not be supported for the type of the 
  /// Python object. For example,
  /// \code
  ///    c = cols(myDict)
  /// \endcode
  /// returns an error because the number of columns is not meaningful
  /// for a dictionary object.

  template <typename From>
  struct ContainerLength {};

  /// Returns the number of elements in the <code>ContainerType</code>
  /// object.
  ///
  /// @param  from          A Python sequence.
  ///
  /// @return The length of the object passed.
  ///
  /// \tparam ContainerType The type of the container.
  template <typename ContainerType>
  int len(const ContainerType &from) {
    return ContainerLength<ContainerType>::len(from);
  }

  /// Returns the number of dimensions of the <code>ContainerType</code>
  /// object. This is typically used for NumPy arrays. This function returns 1 for one-dimensional sequences.
  ///
  /// @param  from          A multi-dimensional Python sequence.
  /// @return The number of dimensions of the Python sequence.
  ///
  /// \tparam ContainerType The type of the container. If a type can never hold a multi-dimensional sequence (e.g. <code>PySetObject*</code>), usually a compiler error results.
  template <typename ContainerType>
  int ndim(const ContainerType &from) {
    return ContainerLength<ContainerType>::ndim(from);
  }

  /// Returns the size of a specific dimension of a multi-dimensional
  /// Python sequence type, such as a NumPy array.
  ///
  /// @param from           A multi-dimensional Python sequence.
  /// @param i              The dimension index.
  ///
  /// @return The size of the dimension.
  ///
  /// \tparam ContainerType If a type can never hold a multi-dimensional sequence (e.g. <code>PySetObject*</code>), usually a compiler error results.
  template <typename ContainerType>
  int dim(const ContainerType &from, int i) {
    return ContainerLength<ContainerType>::dim(from, i);
  }

  /// Returns the number of rows of a two-dimensional Python sequence
  /// type such as a NumPy array.
  ///
  /// @param from              A two-dimensional Python sequence.
  ///
  /// @return The number of rows.
  ///
  /// \tparam ContainerType    The type of the container (e.g. <code>PyArrayObject*</code>). If a type can never hold a two-dimensional sequence (e.g. <code>PySetObject*</code>), usually a compiler error results. Run-time incompatibility results in an exception thrown.
  template <typename ContainerType>
  int rows(const ContainerType &from) {
    return ContainerLength<ContainerType>::rows(from);
  }

  /// Returns the number of columns of a two-dimensional Python sequence
  /// type such as a NumPy array. This method should not be implemented
  /// for <code>ContainerType</code>
  ///
  /// @param from              A two-dimensional Python sequence.
  ///
  /// @return The number of columns.
  ///
  /// \tparam ContainerType    The type of the container (e.g. <code>PyArrayObject*</code>). If a type is not a two-dimensional sequence, usually a compiler error results.
  template <typename ContainerType>
  int cols(const ContainerType &from) {
    return ContainerLength<ContainerType>::cols(from);
  }

  /// A templated specialization for finding the dimensions, length,
  /// number of rows, and number of columns for Python lists.
  template <>
  struct ContainerLength<PyListObject*> {

    /// Returns the length of a Python list.
    ///
    /// @param src     A Python list object.
    ///
    /// @return        The length of the Python list object passed.
    static int len(PyListObject *src) {
      if (!PyList_Check(src)) {
	throw std::string("PyCPP: len() on object expected to be Python list.");
      }
      return PyList_Size((PyObject*)src);
    }

    /// Returns the length of a Python list, passed as a PyObject*.
    ///
    /// @param src     A Python list object passed as a PyObject*.
    ///
    /// @return        The length of the Python list object passed.
    static int len(PyObject *src) {
      return PyList_Size(src);
    }

    /// Returns the number of dimensions of a Python list, which is
    /// always one.
    ///
    /// @param src     A Python list object.
    ///
    /// @return        The number of dimensions (==1).
    static int ndims(PyListObject *src) {
      return 1;
    }

    /// Returns the number of dimensions of a Python list, which is
    /// always one.
    ///
    /// @param src     A Python list object passed as a PyObject*.
    ///
    /// @return        The number of dimensions (==1).
    static int ndims(PyObject *src) {
      return 1;
    }

    /// Returns the dimension size of a dimension of a Python list. The
    /// dimension specifier should always be 0 because Python lists are
    /// one-dimensional.
    ///
    /// @param src     A Python list object.
    ///
    /// @return        The number of dimensions (==1).
    static int dims(PyListObject *src, int i) {
      if (i != 0) {
	throw std::string("PyCPP: dimension specifier to dims() out of bounds on Python list.");
      }
      return len(src);
    }

    /// Returns the dimension size of a dimension of a Python list. The
    /// dimension specifier should always be 0 because Python lists are
    /// one-dimensional.
    ///
    /// @param src     A Python list object passed as a PyObject*.
    ///
    /// @return        The number of dimensions (==1).
    static int dims(PyObject *src, int i) {
      if (i != 0) {
	throw std::string("PyCPP: dimension specifier to dims() out of bounds on Python list");
      }
      return len(src);
    }
  };
}

#endif
