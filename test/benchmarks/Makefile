TEST_ARGS = $(shell echo $@ | perl -pe 's/([^.]*)\.([^.]*)\.([^.]*)\.test/--accelerator=$$1 --test=$$2 --report=$$3/')

TESTS := $(wildcard *.test)
all: $(TESTS)
.PHONY: $(TESTS) all

ifndef V
  QUIET_AGGREGATE    = @echo '  ' AGGREGATE $(TEST_ARGS);
  QUIET_DIFF         = @echo '  ' DIFF $@;
  QUIET_RM           = @echo '  ' RM $@.tmp;
endif

$(TESTS):
	$(QUIET_AGGREGATE)python3 ../../benchmarks/aggregate.py $(TEST_ARGS) \
		--format=csv $(wildcard *.jsonl) > $@.tmp
	$(QUIET_DIFF)git diff --no-index $@ $@.tmp
	$(QUIET_RM)$(RM) $@.tmp

clean:
	$(RM) *.tmp
.PHONY: clean
