#ifndef CONVERTXY_TOON_HPP
#define CONVERTXY_TOON_HPP

#include <TooN/TooN.h>
#include <vector>
#include <sstream>

namespace ConvertXY {

  using namespace TooN;
  using namespace std;

  template <int Rows, int Cols, class ElemType>
  class ObjectFactory<Matrix<Rows, Cols, ElemType, Reference::RowMajor> > {

  public:
    static Matrix<Rows, Cols, ElemType, Reference::RowMajor> create(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      check(dimensions, bufferPtr);
      return Matrix<Rows, Cols, ElemType, Reference::RowMajor>((ElemType*)bufferPtr, dimensions[0], dimensions[1]);
    }


    static Matrix<Rows, Cols, ElemType, Reference::RowMajor>* create_ptr(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      check(dimensions, bufferPtr);
      return new Matrix<Rows, Cols, ElemType, Reference::RowMajor>((ElemType*)bufferPtr, dimensions[0], dimensions[1]);
    }

  private:

    static void check(const vector <size_t> &dimensions, void *bufferPtr = 0) {
      if (dimensions.size() != 2) {
	ostringstream ostr;
	ostr << "ObjectFactory<TooN::Matrix>: dimensions mismatch, " << dimensions.size() << " expected (2)";
	throw Py::TypeError(ostr.str());
      }
      if (Rows != -1 && dimensions[0] != Rows) {
	ostringstream ostr;
	ostr << "ObjectFactory<TooN::Matrix>: row mismatch, " << dimensions[0] << " expected (" << Rows << ")";
	throw Py::TypeError(ostr.str());
      }
      if (Cols != -1 && dimensions[1] != Cols) {
	ostringstream ostr;
	ostr << "ObjectFactory<TooN::Matrix>: column mismatch, " << dimensions[1] << " expected (" << Cols << ")";
	throw Py::TypeError(ostr.str());
      }      
    }
  };
}

#endif
