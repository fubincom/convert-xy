Convert-XY
==========

This SVN repository contains two directories, PyCPP and 2.x. The PyCPP
contains the header files for Convert-XY 1.0 while the 2.x contains
headers for the complete rewrite of Convert-XY. Convert-XY 2.0 is
described in the paper ``Convert-XY: Type-safe Interchange of
Containers Between C++ and Python'' by Damian Eads and Ed Rosten.

Installation
------------

No formal release process is in place yet. Until then, your include
path should point to your SVN check out, which you should update
regularly. configure and Makefile installation scripts are under
development.

Setting your paths
------------------

To use Convert-XY 1.0, set your include path to 1.x, and include
headers prefixed with PyCPP/. To use Convert-XY 2.0, set your
include path to 2.x.
