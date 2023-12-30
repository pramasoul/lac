# lac to-do list

## General
* Solve FIXME's

## Cleanup
* [x] Take out compress_level cruft
  * [x] remove mentions in source
  * [x] remove tests for compresslevel
    * [x] from test_tlacz
  * [x] commit
* [x] mark.skip the ordinality benchmark ramp in tlacz
* [ ] Make logging more sane
  * [ ] Clean up profusion and formats
  * [ ] Make them appropriately enablable perhaps by debug level
  * [x] --log LEVEL command line arg

## Command-line
* [x] Make tlacz -m MODEL feed thru to actual effect
* [x] Make tlacz --device feed thru to actual effect
* [x] Make tlacz --threads feed thru to actual effect
* [ ] Make tlacz -T TEMPERATURE feed thru to actual effect
* [ ] Make tlacz -v feed thru to actual effect
  * [ ] -v to report compression ratio etc
* [ ] Make tlacz -q feed thru to actual effect
* [ ] Make tlacz log_level switch

## Usability
* [x] Make stdout flush often enough to see progress
  * [ ] Debug why too short stdout_chunk_size hangs. Is it footer decoding?
* [ ] Review file naming and overwriting and correct
* [ ] Provide means of establishing defaults for
  * [ ] model
  * [ ] device
  * [ ] threads
* [ ] Establish return values when run as a command


## Robustness
* [ ] Use `model.config` info e.g. GPTConfig(block_size=1024, vocab_size=50304,...)`
* [ ] Make asserts have reasonable messages

## Format
* [ ] Fix header to use lac's informative header
* [ ] Make switch for one-bit header mode
  * [ ] first bit==0 -> no more header
  * [ ] first bit==0 -> no more header
* [ ] Make footer have hash check of correct results

## Testing
* [ ] Make compression-data for tests as importable data file that rebuilds
* [ ] Test with other models

## Lurking Trouble
* [x] Fix idx endless growth in prediction service

## Maintainability & Presentation
* [ ] Map out the code structure
* [ ] Refactor & consolidate
* [ ] Sphinx document the code

## Features Development
* [ ] Load models from HF or other sources
* [ ] Option to compress in blocks for parallelism and ripstop

## Experiments
* [ ] Consistency of logits
  * [ ] across environments
  * [ ] across modes
  * [ ] when batching
  * [ ] when extending idx

## Ideas & Research
* [ ] Dealing with noise in pdf
  * [ ] Quantization
  * [ ] Spacing - "trees"
  * [ ] Backtracking weighted on proximity to quantization boundary