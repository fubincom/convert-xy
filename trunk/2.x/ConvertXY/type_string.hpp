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

#ifndef CONVERTXY_TYPE_STRING_HPP
#define CONVERTXY_TYPE_STRING_HPP

namespace ConvertXY {

  using namespace std;

  template <class T, int Layer = 1>
  struct TypeString {
    static const string str() {
      typedef TypeString<T, Layer - 1> LowerLayer;
      return LowerLayer::str();
    }
  };

  template <>
  template <class T>
  struct TypeString<T, 0> {
    static const string str() {
      return "T";
    }
  };

  template <int I>
  struct TypeIntString {
    static const string str() {
      ostringstream ostr;
      ostr << I;
      return ostr.str();
    }
  };

#define CONVERTXY_DEFINE_SIMPLE_TYPE_STRING(t, s) \
  template <int Layer> \
  struct TypeString<t, Layer> { \
    static const string str() { \
      return s; \
    } \
  };

#define CONVERTXY_DEFINE_SIMPLE_TYPE_STRING_LAYER(t, s, k) \
  template <> \
  struct TypeString<t, k> { \
    static const string str() { \
      return s; \
    } \
  };


#define CONVERTXY_DEFINE_TYPE_STRING_T(t, s) \
  template <> \
  template <int Layer, class T1> \
  struct TypeString<t<T1>, Layer > { \
    static const string str() { \
      ostringstream ostr; \
      typedef TypeString<T1> T1String; \
      ostr << s << "<" << T1String::str() << " >"; \
      return ostr.str(); \
    } \
  };

#define CONVERTXY_DEFINE_TYPE_STRING_I(t, s) \
  template <> \
  template <int Layer, int T1> \
  struct TypeString<t<T1>, Layer > { \
    static const string str() { \
      ostringstream ostr; \
      ostr << s << "<" << T1 << " >"; \
      return ostr.str(); \
    } \
  };


#define CONVERTXY_DEFINE_TYPE_STRING_TT(t, s) \
  template <> \
  template <int Layer, class T1, class T2> \
  struct TypeString<t<T1, T2>, Layer > { \
    static const string str() { \
      ostringstream ostr; \
      typedef TypeString<T1> T1String; \
      typedef TypeString<T2> T2String; \
      ostr << s << "<" << T1String::str() << "," << T2String::str() << " >"; \
      return ostr.str(); \
    } \
  };

#define CONVERTXY_DEFINE_TYPE_STRING_II(t, s) \
  template <> \
  template <int Layer, int T1, int T2> \
  struct TypeString<t<T1, T2>, Layer > { \
    static const string str() { \
      ostringstream ostr; \
      ostr << s << "<" << T1 << "," << T2 << " >"; \
      return ostr.str(); \
    } \
  };

#define CONVERTXY_DEFINE_TYPE_STRING_TT(t, s) \
  template <> \
  template <int Layer, class T1, class T2> \
  struct TypeString<t<T1, T2>, Layer > { \
    static const string str() { \
      ostringstream ostr; \
      typedef TypeString<T1> T1String; \
      typedef TypeString<T2> T2String; \
      ostr << s << "<" << T1String::str() << "," << T2String::str() << " >"; \
      return ostr.str(); \
    } \
  };
#define CONVERTXY_DEFINE_TYPE_STRING_TTT(t, s) \
  template <> \
  template <int Layer, class T1, class T2, class T3> \
  struct TypeString<t<T1, T2, T3>, Layer > { \
    static const string str() { \
      ostringstream ostr; \
      typedef TypeString<T1> T1String; \
      typedef TypeString<T2> T2String; \
      typedef TypeString<T3> T3String; \
      ostr << s << "<" << T1String::str() << "," << T2String::str() << "," << T3String::str() << " >"; \
      return ostr.str(); \
    } \
  };


#define CONVERTXY_DEFINE_TYPE_STRING_ITT(t, s) \
  template <> \
  template <int Layer, int T1, class T2, class T3> \
  struct TypeString<t<T1, T2, T3>, Layer > { \
    static const string str() { \
      ostringstream ostr; \
      typedef TypeString<T2> T2String; \
      typedef TypeString<T3> T3String; \
      ostr << s << "<" << T1 << "," << T2String::str() << "," << T3String::str() << " >"; \
      return ostr.str(); \
    } \
  };

#define CONVERTXY_DEFINE_TYPE_STRING_IITT(t, s) \
  template <> \
  template <int Layer, int T1, int T2, class T3> \
  struct TypeString<t<T1, T2, T3>, Layer > { \
    static const string str() { \
      ostringstream ostr; \
      typedef TypeString<T3> T3String; \
      ostr << s << "<" << T1 << "," << T2 << "," << T3String::str() << " >"; \
      return ostr.str(); \
    } \
  };

  template <class T, int Layer>
  struct NilZip {
    static const string str() {
      return ", " + TypeString<T, Layer>::str();
    }

    static const string first_str() {
      return TypeString<T, Layer>::str();
    }
  };

#define CONVERTXY_DEFINE_TYPE_STRING_TTTTTT_NIL_ZIP(t, s) \
  template <> \
  template <int Layer, class T1, class T2, class T3, class T4, class T5, class T6> \
  struct TypeString<t<T1, T2, T3, T4, T5, T6>, Layer > { \
    static const string str() { \
      ostringstream ostr; \
      typedef NilZip<T1, Layer> T1String; \
      typedef NilZip<T2, Layer> T2String; \
      typedef NilZip<T3, Layer> T3String; \
      typedef NilZip<T4, Layer> T4String; \
      typedef NilZip<T5, Layer> T5String; \
      typedef NilZip<T6, Layer> T6String; \
      ostr << s << "<" << T1String::first_str() << T2String::str() << T3String::str() << T4String::str() << T5String::str() << T6String::str() << " >"; \
      return ostr.str(); \
    } \
  };
}

#endif
