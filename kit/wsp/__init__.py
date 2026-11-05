"""wsp-kit: weight-space programming you can run.

Version 0.1.1. The public interface is the CLI (``wsp demo``) and the MCP
tools in ``wsp.mcp_server``. Report format version 1 (see ``wsp.report``)
is stable: later versions add fields, they do not rename or remove
existing ones, so harnesses built against v1 keep working.
"""

__version__ = "0.1.1"
REPORT_VERSION = 1
