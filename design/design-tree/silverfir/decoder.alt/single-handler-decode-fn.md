- Function code is decoded by a free function that takes exactly one consumer
  by generic reference (`decode_function`): the single pass walks the code
  bytes and, for each instruction, calls back into that one handler through the
  handler trait. No materialized instruction vector exists between decode and
  consumer.

- A consumer that needs a second observer of the same walk owns it directly: a
  consumer holds the other handler as a field and forwards each opcode to it by
  hand from inside its own callback.

## Moves

- 2024-02-01 (5bb02079) replaced by [[decoder]]: a single-consumer decode
  function cannot express several independent observers of one decode walk — it
  forces one consumer to embed and hand-forward to the others; a decoder object
  holding a vector of handlers fans every opcode out to all registered
  observers, so validator and printer become sibling handlers on one walk
  (diff).
