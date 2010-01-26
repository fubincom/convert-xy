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

/// \file   CVDSTL.hpp
///
/// \brief This header is included to convert between NumPy/Python
/// containers and LIBCVD constructs, and assumes any header in
/// STL can be included.
///
/// \author Damian Eads

#ifndef CONVERTXY_CVDSTL_HPP
#define CONVERTXY_CVDSTL_HPP

namespace ConvertXY {

  using namespace std;
  using namespace CVD;

  template <>
  template <class ElemType>
  struct ConvertToCPP<Image<ElemType>, PyIteratorProtocol, AllocateCopy<> >
    : public ConvertToCPPBase<Image<ElemType>, AllocateCopy<> > {

    ConvertToCPP() {}
    virtual ~ConvertToCPP() {}
    
    virtual void convert(PyObject *src, Image<ElemType> &dst) const {
      vector<vector<ElemType> > vrep;
      //ConvertXY::ConvertToCPP<vector<vector<ElemType> >, PyIteratorProtocol, Copy<> > converter;
      //converter.convert(src, vrep);
      Py::Object _src(src);
      ConvertXY::convert(src, vrep);
      const size_t max_y = vrep.size();
      size_t max_x = 0;
      for (size_t y = 0; y < max_y; y++) {
	const size_t xsz = vrep[y].size();
	if (xsz > max_x) {
	  max_x = xsz;
	}
      }
      if (max_x == 0 || max_y == 0) {
	return;
      }
      dst.resize(ImageRef(max_x, max_y));
      for (size_t y = 0; y < max_y; y++) {
	for (size_t x = 0; x < vrep[y].size(); x++) {
	  dst[y][x] = vrep[y][x];
	}
      }
    }

    virtual bool isImplemented() const { return true; }

  };
}

#endif
