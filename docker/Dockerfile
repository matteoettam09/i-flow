FROM tensorflow/tensorflow:latest-py3-jupyter

RUN pip install tensorflow==2.0.0b0
RUN pip install matplotlib

ENV DEBIAN_FRONTEND noninteractive

RUN echo 'path-exclude /usr/share/doc/*\npath-include /usr/share/doc/*/copyright\npath-exclude /usr/share/lintian/*' > /etc/dpkg/dpkg.cfg.d/01_nodoc; \
apt-get update; \
apt-get upgrade -q -y; \
apt-get -q -y --no-install-recommends install dkms lubuntu-core evince zile wget automake g++ gcc gfortran gsl-bin libgsl0-dev libtool make subversion git swig ssh zlib1g-dev texlive texlive-pstricks graphicsmagick graphicsmagick-imagemagick-compat screen graphviz gv sqlite3 libsqlite3-dev libmpich-dev mpich scons; \
apt-get -q -y purge adwaita-icon-theme humanity-icon-theme texlive-latex-recommended-doc texlive-pstricks-doc texlive-latex-base-doc texlive-fonts-recommended-doc; \
apt-get -q -y --no-install-recommends install texlive-latex-extra texlive-metapost feynmf vim less; \
apt autoremove; \
apt-get autoclean; \
rm -rf /var/lib/apt/lists /var/cache/apt; \
rm -rf $(find / -name doc 2>&1 | grep -v Permission)

RUN mkdir rivet; \
cd rivet/; \
wget https://phab.hepforge.org/source/rivetbootstraphg/browse/2.7.2/rivet-bootstrap?view=raw -O rivet-bootstrap; \
chmod +x rivet-bootstrap; \
INSTALL_PREFIX=/usr/local ./rivet-bootstrap; \
cd ..; \
rm -rf rivet

RUN wget https://lhapdf.hepforge.org/downloads/?f=LHAPDF-6.2.3.tar.gz -O LHAPDF-6.2.3.tar.gz; \
tar -xzf LHAPDF-6.2.3.tar.gz; \
cd LHAPDF-6.2.3; \
./configure --prefix=/usr/local; \
make; \
make install; \
cd ..; \
rm -rf LHAPDF-6.2.3*
RUN wget http://lhapdfsets.web.cern.ch/lhapdfsets/current/CT14nlo.tar.gz -O- | tar xz -C /usr/local/share/LHAPDF

RUN cd /usr/local; \
git clone https://gitlab.com/openloops/OpenLoops.git; \
cd OpenLoops; \
./scons
RUN cd /usr/local/OpenLoops; ./openloops libinstall ppll pptt

RUN git clone -b rel-2-2-7 https://gitlab.com/sherpa-team/sherpa.git; \
cd sherpa; \
wget http://www.slac.stanford.edu/~shoeche/pub/sherpa-2.2.7_mlps.patch; \
patch -p1 < sherpa-2.2.7_mlps.patch; \
autoreconf -fi; \
./configure --prefix=/usr/local --enable-lhapdf=/usr/local --enable-fastjet=/usr/local --enable-hepmc2=/usr/local --enable-rivet=/usr/local --enable-openloops=/usr/local/OpenLoops --enable-analysis --enable-pyext --enable-mpi --enable-gzip CC=mpicc CXX=mpic++; \
make -j2 CXXFLAGS="-O2 -std=c++11 -DDEBUG__BG" AM_CXXFLAGS="-O2 -std=c++11 -DDEBUG__BG"; \
make install; \
mkdir /usr/local/Sherpa; \
mv Examples /usr/local/Sherpa; \
cd ..; \
rm -rf sherpa

RUN pip install mpi4py
RUN pip install tensorflow_probability

RUN echo "export PATH=/usr/local/bin:\$PATH\n\
export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH\n\
export PYTHONPATH=\$PYTHONPATH:/usr/local/lib/python3.6/site-packages\n\
. /usr/local/rivetenv.sh\n\
if [ "\$PS1" ]; then PS1='\\h:\\w\\$ '; fi" > /etc/bash.bashrc

ENTRYPOINT ["/bin/bash"]