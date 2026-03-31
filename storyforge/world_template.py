"""
World definition template for StoryForge.

Provides the default template for world.md files that define persistent
story universe elements (characters, places, lore) included verbatim
in every story generation prompt.
"""

WORLD_FILENAME = "world.md"

WORLD_TEMPLATE = """\
# Story World

<!-- This file defines your story universe. Everything here is included
     verbatim in every story prompt, so keep it focused and concise.
     Delete these HTML comments once you've filled in your world. -->

## Characters

<!-- Describe your recurring characters here. Include visual details
     for consistent illustration. Example: -->

<!-- ### Luna
A curious 7-year-old girl with curly red hair and bright green eyes.
She always wears her favorite purple rain boots, even on sunny days.
Brave but sometimes impulsive — she leaps before she looks. -->

## Places

<!-- Describe important locations in your stories. Example: -->

<!-- ### The Whispering Woods
A magical forest behind Luna's house where the trees murmur secrets
at twilight. Paths shift when no one is looking, and friendly fireflies
guide lost travelers home. -->

## Rules & Lore

<!-- Any world rules, magic systems, or recurring elements. Example: -->

<!-- - Animals can talk, but only to children who believe.
- The old lighthouse grants one wish per year on the summer solstice. -->

## Relationships

<!-- How characters relate to each other. Example: -->

<!-- - Luna and Max are best friends who met in the Whispering Woods.
- Professor Owlsworth is Luna's reluctant mentor. -->

## Tone & Style Notes

<!-- Any notes about the overall feel of your story world. Example: -->

<!-- - Stories should feel cozy and safe, even during adventures.
- Humor comes from characters, not sarcasm.
- Magic is wondrous, never scary. -->
"""
