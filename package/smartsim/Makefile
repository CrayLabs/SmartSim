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
# Parent directory
BUILDIR := $(shell cd ../;pwd)

# Names
# Assume BUILDIR is .../name/version and extract the fields
NAME := $(shell perl -e '"$(BUILDIR)" =~ /^\S+\/(.+)\/(.+)$$/; print $$1')
VERSION := $(shell perl -e '"$(BUILDIR)" =~ /^\S+\/(.+)\/(.+)$$/; print $$2')
RELEASE = 1

# Staging directory
PREFIX = /opt/cray/$(NAME)/$(VERSION)
BUILD_ROOT = /tmp/$(USER)/$(NAME)-$(VERSION)-$(RELEASE)

# Misc
RPMBUILD = rpmbuild
ifdef TARGET_CPU
  RPM_FLAGS = --target $(TARGET_CPU)
endif

.PHONY: RPM
RPM: $(NAME).spec $(NAME)-modulefile Makefile
	mkdir -p $(BUILDIR)/BUILD
	$(RPMBUILD) -bb $(NAME).spec $(RPM_FLAGS)

# Generate spec file from template
$(NAME).spec: template.spec Makefile
	sed -e 's,%SVNHOME%,$(BUILDIR),g' \
	    -e 's,%BUILD_ROOT%,$(BUILD_ROOT),g' \
	    -e 's,%PREFIX%,$(PREFIX),g' \
            -e 's,%NAME%,$(NAME),g' \
            -e 's,%VERSION%,$(VERSION),g' \
            -e 's,%RELEASE%,$(RELEASE),g' \
            -e 's,%SOURCE_DIR%,$(SOURCE_DIR),g' \
            -e 's,%URL%,$(URL),g' \
	   $< >$@

# Generate module file from template
$(NAME)-modulefile: template-modulefile Makefile
	sed -e 's,setenv \[product\],setenv $(subst -,_,$(NAME)),' \
	    -e 's,\[product\],$(NAME),' \
            -e 's,\[product-version\],$(VERSION),' \
            -e 's,\[prefix\],$(PREFIX),' \
	   $< >$@

# Clean
.PHONY: clean
clean:
	rm -f $(NAME).spec
	rm -f $(NAME)-modulefile
	rm -rf $(BUILDIR)/BUILD
	rm -rf $(BUILD_ROOT)
