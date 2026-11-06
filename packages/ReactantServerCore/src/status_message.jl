# The text a gRPC status carries back to a client, shared by the worker and the gateway. It leaves in
# the `grpc-message` response header, so it must be one line of valid text, and it must not be
# mistaken for another header: gRPCClient finds a call's status by searching its response headers for
# `grpc-status: <n>` anywhere, the message included (GRPC-2), so a message that echoes client text
# containing that pattern (an input name, a region name, an exception message) could turn an error
# into an OK with an empty body.

"""
    sanitize_status_message(msg) -> String

`msg` made safe to send as a gRPC status message: `grpc-status:` and `grpc-message:` (in any case)
become `grpc-status=` and `grpc-message=`, so no client can read them as its call's status; control
characters become spaces, since a header value is one line; and bytes that are not valid UTF-8
become U+FFFD. Every status message the worker and the gateway send passes through it.
"""
function sanitize_status_message(msg::AbstractString)
    s = String(msg)
    isvalid(s) || (s = String(map(c -> isvalid(c) ? c : '�', collect(s))))
    s = replace(s, r"[\x00-\x1f\x7f]" => " ")
    return replace(s, r"(?i)(grpc-(?:status|message))(\s*):" => s"\1\2=")
end

"""
    error_request_id(id="") -> String

The request id an `INTERNAL` error is reported and logged under: the client's own KServe request
`id` when it sent one (sanitized and trimmed, at most 64 characters), otherwise 8 random hex digits.
"""
function error_request_id(id::AbstractString = "")
    s = String(strip(sanitize_status_message(id)))
    isempty(s) && return string(rand(UInt32); base = 16, pad = 8)
    return length(s) <= 64 ? s : first(s, 64)
end
