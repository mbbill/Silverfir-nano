- Top-of-stack lanes (t0–t3) and a depth byte are passed by value as scalar
  arguments to every handler; each handler returns a `Next` bundle (next
  instruction pointer plus updated lanes) by value.
