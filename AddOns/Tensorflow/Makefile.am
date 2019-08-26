if ENABLE_TENSORFLOW

pkglib_LTLIBRARIES = libSherpaTF.la

libSherpaTF_la_SOURCES = Interface.C
libSherpaTF_la_CPPFLAGS= @TF_CFLAGS@ -O3 -std=c++11
libSherpaTF_la_LDFLAGS = -shared @TF_LFLAGS@ -fPIC
libSherpaTF_la_LIBADD  = -L@SHERPABUILDDIR@/Main -lSherpaMain -L@MODELBUILDDIR@/SM -lSherpaSM -L@PDFBUILDDIR@/Electron -lPDFESherpa @ATOOLSLIBS@ 

bin_PROGRAMS  =	Foam

Foam_SOURCES        =	Foam_Interface.C Foam.C

Foam_LDADD  =	-L@SHERPABUILDDIR@/Main -lSherpaMain \
		-L@SHERPABUILDDIR@/Single_Events -lSherpaSingleEvents \
		@ATOOLSLIBS@

endif

