TEST_ARGS = $(shell echo $@ | perl -pe 's/([^.]*)\.([^.]*)\.([^.]*).*\.test/--accelerator=$$1 --test=$$2 --report=$$3/')
EMBEDDED_TEST_ARGS = $(shell cat $@ | grep '^# ARGS: ' | perl -pe 's/^# ARGS: (.*)/$$1/')

TESTS := $(wildcard *.test)
all: $(TESTS)
.PHONY: $(TESTS) all

ifndef V
  QUIET_AGGREGATE    = @echo '  ' AGGREGATE $(TEST_ARGS) $(EMBEDDED_TEST_ARGS);
  QUIET_DIFF         = @echo '  ' DIFF $@;
  QUIET_RM           = @echo '  ' RM $@.tmp;
endif

$(TESTS):
	$(QUIET_AGGREGATE)python3 ../../benchmarks/aggregate.py \
		--format=csv \
		$(TEST_ARGS) $(EMBEDDED_TEST_ARGS) \
		$(wildcard *.jsonl) > $@.tmp
	$(QUIET_DIFF)git diff -I'^# ARGS: ' --no-index $@ $@.tmp
	$(QUIET_RM)$(RM) $@.tmp

clean:
	$(RM) *.tmp
.PHONY: clean
