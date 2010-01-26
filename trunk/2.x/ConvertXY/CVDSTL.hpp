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
