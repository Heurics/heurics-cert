# ccpocheck -- independent C verifier for the CCPO dual-bound certificate.
#
# `-ffp-contract=off` is not optional: an FMA fuses a multiply and an add into one rounding, and the
# directed-rounding analysis in ccpocheck.c assumes one rounding per operation with a known
# direction. `-fno-fast-math` for the same reason -- fast-math permits reassociation, which makes
# every enclosure in this file a guess. The binary re-checks at runtime that nextafter still moves.
CC      ?= cc
CFLAGS  ?= -std=c99 -O2 -Wall -Wextra -ffp-contract=off -fno-fast-math
LDLIBS  := -lm

ccpocheck: ccpocheck.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

clean:
	rm -f ccpocheck

.PHONY: clean
