"""wsp-kit: weight-space programming you can run.

Version 0.2.0. The public interface is the CLI (``wsp demo`` for the
kit, ``wsp run`` for any open-weight model, ``wsp status`` for one
line and ``wsp serve`` for the local status page) and the MCP tools
in ``wsp.mcp_server``. Report format version 1 (see ``wsp.report``)
is stable: later versions add fields, they do not rename or remove
existing ones, so harnesses built against v1 keep working.
"""

__version__ = "0.2.0"
REPORT_VERSION = 1
