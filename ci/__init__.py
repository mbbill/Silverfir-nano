"""Silverfir Nano CI implementation.

The modules in this package are intentionally CI-specific. Local development
uses ordinary Cargo commands; GitHub Actions invokes these entry points for
the exhaustive host, cross-runtime, bare-metal, lint, and performance gates.
"""
