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

#ifndef SCTL_HOLDER_HPP
#define SCTL_HOLDER_HPP

/// \file   dispatch.hpp
///
/// \brief  This header contains definitions for place holder
///         template classes.
///
/// \author Damian Eads

namespace PyCPP {

  /// When using the
  /// \code
  ///    convert(x, y);
  /// \endcode
  /// idiom to convert a Python object <code>x</code> to a c++ value object
  /// <code>y</code>, <code>y</code> must be constructed prior to 
  /// calling <code>convert</code>. This is problematic when <code>y</code>'s
  /// class does not have a default constructor. For example, the TooN
  /// reference array type has no default constructor so
  /// \code
  ///    Matrix<Dynamic, Dynamic, float, Reference::RowMajor> y;
  ///    convert(x, y)
  /// \endcode
  /// will result in a compiler error. This class serves as a place-holder
  /// for an object. When it loses scope, it invokes the destructor of
  /// the object inside of it.
  /// \code
  ///    Holder<Matrix<Dynamic, Dynamic, float, Reference::RowMajor>> y;
  ///    convert(x, y);
  ///    cout << "The matrix is: " << y.getReference();
  /// \endcode


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
	throw std::string("error: Holder not initialized!");
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
	throw std::string("error: Holder not initialized!");
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
}
