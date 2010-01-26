// Python/C++ Interface Template Library
//
// Copyright (C) 2009 Damian Eads

#ifndef CONVERTXY_HOLDER_HPP
#define CONVERTXY_HOLDER_HPP

#include "converter.hpp"

/// \file   dispatch.hpp
///
/// \brief  This header contains definitions for place holder
///         template classes.
///
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

  template <class T, class BA>
  struct DefaultToCPPConvertAction<Holder<T>, BA> {
    typedef DefaultToCPPConvertAction<T, BA> HeldClass;
    typedef typename HeldClass::Action HeldAction;
    typedef Allocate<HeldAction> Action;
  };


  template <class PyType>
  template <class HeldType, class HeldAction>
  struct ConvertToCPP<Holder<HeldType>, PyType, Allocate<HeldAction> >
    : public ConvertToCPPBase<Holder<HeldType>, Allocate<HeldAction> > {

    ConvertToCPP() {}
    
    void convert(PyObject *src, Holder<HeldType> &dst) {
      ConvertToCPPBase<HeldType, HeldAction> &converter(ToCPPDispatch<HeldType, HeldAction>::getConverter(src));
      vector <size_t> dimensions(converter.getSize(src));
      void *buffer_ptr = converter.getBufferPointer(src);
      dst.own(ObjectFactory<HeldType>::create_ptr(dimensions, buffer_ptr));
      converter.convert(dst[dst.size()-1], dst.getReference());
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
