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

#ifndef CONVERTXY_HOLDER_HPP
#define CONVERTXY_HOLDER_HPP

#include "converter.hpp"

/// \file   holder.hpp
///
/// \brief  This header defines the Holder class, useful for deferred.
///         construction of objects without default constructors or
///         objects of classes with fields that immutable after construction.
/// \author Damian Eads

namespace ConvertXY {

  template <class T>
  class Holder {

  public:

    /// Constructs a place holder object without an object stored
    /// inside of it.
    Holder() : val(0) {}

    /// If an object is held by this place holder object, its
    /// destructor is invoked.
    ~Holder() {
      delete val;
    }

    /// Returns a reference to the object stored in this place
    /// holder. An exception is thrown if it does not exist.
    ///
    /// @return A reference to the object stored in this place
    ///         holder.
    ///
    /// \exception std::string If no object exists in this place
    ///                        holder.
    T &getReference() {
      if (val == 0) {
	throw Py::Exception("Holder: not initialized!");
      }
      return *val;
    }

    /// Returns a reference to the object stored in this place
    /// holder. An exception is thrown if it does not exist.
    ///
    /// @return A reference to the object stored in this place
    ///         holder.
    ///
    /// \exception std::string If no object exists in this place
    ///                        holder.
    const T &getReference() const {
      if (val == 0) {
	throw Py::Exception("Holder: not initialized!");
      }
      return *val;
    }

    /// Returns a reference to the object stored in this place
    /// holder. An exception is thrown if it does not exist.
    ///
    /// @return A reference to the object stored in this place
    ///         holder.
    ///
    /// \exception std::string If no object exists in this place
    ///                        holder.
    T &operator()() {
      if (val == 0) {
	throw Py::Exception("Holder: not initialized!");
      }
      return *val;
    }

    /// Returns a reference to the object stored in this place
    /// holder. An exception is thrown if it does not exist.
    ///
    /// @return A reference to the object stored in this place
    ///         holder.
    ///
    /// \exception std::string If no object exists in this place
    ///                        holder.
    const T &operator()() const {
      if (val == 0) {
	throw Py::Exception("Holder: not initialized!");
      }
      return *val;
    }


    /// Assigns ownership of an object to this place holder. When
    /// the place holder is destructed, the object passed will be
    /// destructed. If <code>own</code> was previously invoked,
    /// the place holder will no longer own the previous object
    /// stored inside it.
    ///
    /// @param obj    The object for this place holder to own.
    void own(T *obj) {
      val = obj;
    }

  private:

    /// A pointer to the object stored in this place holder.
    T *val;
  };

  CONVERTXY_DEFINE_TYPE_STRING_T(Holder, "Holder");

  template <class T, class BA>
  struct DefaultToCPPConvertAction<Holder<T>, BA> {
    typedef DefaultToCPPConvertAction<T, BA> HeldClass;
    typedef typename HeldClass::Action HeldAction;
    typedef Allocate<HeldAction> Action;
  };


  //template <class PyType>

  //template <>
  template <>
  template <class HeldType, class PyType, class HeldAction>
  struct ConvertToCPP<Holder<HeldType>, PyType, Allocate<HeldAction> >
    : public ConvertToCPPBase<Holder<HeldType>, Allocate<HeldAction> > {

    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}
    
    virtual void convert(PyObject *src, Holder<HeldType> &dst) const {
      //HeldAction h(5);
      cerr << "Holder!" << endl;
      ConvertToCPPBase<HeldType, HeldAction> &converter(ToCPPDispatch<HeldType, HeldAction>::getConverter(src));
      vector <size_t> dimensions(converter.getSize(src));
      void *buffer_ptr = converter.getBufferPointer(src);
      dst.own(ObjectFactory<HeldType>::create_ptr(dimensions, buffer_ptr));
      converter.convert(src, dst.getReference());
    }
  };

  template <class HeldType, class DefaultBufferAction>
  struct DefaultToPythonStructure<Holder<HeldType>, DefaultBufferAction> {
    typedef DefaultToPythonStructure<HeldType, DefaultBufferAction> HeldClass;
    typedef typename HeldClass::Structure Structure;
  };

  template <class HeldType, class HeldStructure>
  struct ConvertToPython<Holder<HeldType>, HeldStructure > {
    static PyObject* convert(const Holder<HeldType> &src) {
      return ConvertToPython<HeldType, HeldStructure>::convert(src);
    }
  };
}

#endif
