"""Resolving native drug identifiers to PubChem CIDs.

``combinations`` recognises fixed-ratio combination therapies from their name and
describes their components; ``overrides`` holds the curated name-to-CID table for
compounds PubChem cannot resolve from the name the source dataset uses;
``names`` fills in a display name for drugs PubChem could not name.
"""
