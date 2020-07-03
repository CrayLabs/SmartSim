#
# Copyright 2018 Cray Inc. All Rights Reserved.
#
# (c) Cray Inc.  All Rights Reserved.  Unpublished Proprietary
# Information.  This unpublished work is protected by trade secret,
# copyright and other laws.  Except as permitted by contract or
# express written permission of Cray Inc., no part of this work or
# its content may be used, reproduced or disclosed in any form.
#
#
%define _topdir %SVNHOME%
%define debug_package %{nil}

Summary: %NAME%
Name: %NAME%
Version: %VERSION%
Release: %RELEASE%
Group: Development
Packager: Cray Inc
Vendor: Cray
License: Cray
Url: %URL%

Source: %{name}-%{version}.tar.gz
Prefix: %PREFIX%
AutoReqProv: no

Buildroot: %BUILD_ROOT%

%define _python_bytecompile_errors_terminate_build 0
%global _missing_build_ids_terminate_build %{nil}
%define __jar_repack %{nil}

%description
%NAME%

%prep
if [ ! $RPM_BUILD_ROOT == "/" ]; then
    rm -rf $RPM_BUILD_ROOT
fi

%setup -T -c
install_dir=%SOURCE_DIR%
CP="cp --preserve=timestamps"


%build
echo No build necessary.

%install
CP="cp --preserve=timestamps"

SMARTSIM_BASE_VERSION=$(echo %{version} | perl -p -e 's,(\d+\.\d+\.\d+).*,$1,g')
SMARTSIM_DIR=%{prefix}
mkdir -p $RPM_BUILD_ROOT/$SMARTSIM_DIR
rsync -a --exclude '*.sh' $SOURCE_DIR/dist $RPM_BUILD_ROOT/$SMARTSIM_DIR

cd $RPM_BUILD_ROOT/$SMARTSIM_DIR

tar -xzf dist/SmartSim-*.tar.gz
# Move module into current directory
rsync -a $RPM_BUILD_ROOT/$SMARTSIM_DIR/SmartSim-*/* $RPM_BUILD_ROOT/$SMARTSIM_DIR

rm -f $(find $RPM_BUILD_ROOT -name '*~')


# module file
modulefile_dir=$RPM_BUILD_ROOT/%{prefix}/../../modulefiles/%NAME%
mkdir -p $modulefile_dir
$CP $RPM_SOURCE_DIR/../SPECS/%NAME%-modulefile $modulefile_dir/%{version}

%files
%defattr(-, root, root)
%{prefix}
%{prefix}/../../modulefiles/%NAME%/*

%pre

%post
# default symlink
if [[ -z "$NO_DEFAULT" && -z "$DO_NOT_INSTALL_AS_DEFAULT" ]] ; then
  [ ! -L /opt/cray/%NAME%/default ] || unlink /opt/cray/%NAME%/default
  ln -sf %{version} /opt/cray/%NAME%/default
fi

%preun
# Remove default link if it points to this version
if [ "$(readlink /opt/cray/%NAME%/default)" == %{version} ] ; then
  unlink /opt/cray/%NAME%/default
fi
