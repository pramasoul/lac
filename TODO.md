# lac to-do list

## General
* [ ] Solve FIXME's
* [ ] ~~Change to text mode rather than bzip2's binary~~

## Cleanup
* [x] Take out compress_level 1-9 cruft
  * [x] remove mentions in source
  * [x] remove tests for compresslevel
    * [x] from test_tlacz
  * [x] commit
* [x] mark.skip the ordinality benchmark ramp in tlacz
* [ ] Make logging more sane
  * [ ] Clean up profusion and formats
  * [ ] Make them appropriately enablable perhaps by debug level
  * [x] --log LEVEL command line arg
* [ ] Replace mentions of bzip2 and gzip
* [ ] Rationalize defaults in function definitions
* [ ] Remove spurious `ctx`
* [ ] Does LACTokDecompressor still need token_buffer?
* [ ] Clean up the LACTokDecompressor save_toks?

## Command-line
* [x] Make tlacz -m MODEL feed thru to actual effect
* [x] Make tlacz --device feed thru to actual effect
* [x] Make tlacz --threads feed thru to actual effect
* [x] Make tlacz -T TEMPERATURE feed thru to actual effect
* [x] Make tlacz -v feed thru to actual effect
  * [ ] -v to report compression ratio etc
* [ ] Make tlacz -q feed thru to actual effect
* [x] Make tlacz log_level switch
* [x] Make -o --output to set output file (see brotli -h)

## Usability
* [x] Make stdout flush often enough to see progress
  * [ ] Debug why too short stdout_chunk_size hangs. Is it footer decoding?
* [ ] Review file naming and overwriting and correct
* [ ] Provide means of establishing defaults for
  * [ ] model
  * [ ] device
  * [ ] threads
* [ ] Establish return values when run as a command
* [x] On decompression, get parameters (model, cpu/cuda, etc) from header
* [x] Make header decompression not require particular numbered cuda device

## Performance
* [ ] Use kv cache
* [ ] Profile

## Robustness
* [ ] Use `model.config` info e.g. GPTConfig(block_size=1024, vocab_size=50304,...)`
* [ ] Make asserts have reasonable messages

## Format
* [ ] Fix header to use lac's informative header
* [ ] Make switch for one-bit header mode
  * [ ] first bit==0 -> no more header
  * [ ] first bit==0 -> no more header
* [ ] Header contain model hash
* [ ] Header contain how to get model
* [x] Make footer have hash check of correct results

## Testing
* [ ] Make compression-data for tests as importable data file that rebuilds
* [ ] Test with other models
* [x] Test as a shell-runnable command
* [ ] Make a script to run all the relevant tests

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
* [ ] Compression vs model & temperature



## Ideas & Research
* [ ] Dealing with noise in pdf
  * [ ] Quantization
  * [ ] Spacing - "trees"
  * [ ] Backtracking weighted on proximity to quantization boundary
