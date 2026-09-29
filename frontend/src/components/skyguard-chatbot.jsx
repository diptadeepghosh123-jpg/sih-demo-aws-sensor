import { useEffect, useRef, useState } from "react";
import { Bot, MessageCircle, Send, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { sendChatMessage } from "@/api";

const initialMessages = [{ role: "assistant", text: "Hello. I am SkyGuard Assistant. Ask me about sensors, anomalies, health, predictions, or the network." }];

export default function SkyGuardChatbot({ sensorId = null }) {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState(initialMessages);
  const [sending, setSending] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    if (open) endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, open]);

  const submit = async (event) => {
    event?.preventDefault();
    const message = input.trim();
    if (!message || sending) return;
    setMessages((current) => [...current, { role: "user", text: message }]);
    setInput("");
    setSending(true);
    try {
      const response = await sendChatMessage(message, sensorId);
      setMessages((current) => [...current, { role: "assistant", text: response?.reply || "I could not produce a response." }]);
    } catch (error) {
      setMessages((current) => [...current, { role: "assistant", text: `Chat service error: ${error.message}` }]);
    } finally {
      setSending(false);
    }
  };

  return (
    <>
      {open && (
        <section className="fixed bottom-20 right-4 z-[1200] flex h-[min(560px,calc(100dvh-6rem))] w-[min(380px,calc(100vw-2rem))] flex-col overflow-hidden rounded-md border border-border bg-surface shadow-panel">
          <header className="flex h-12 shrink-0 items-center justify-between border-b border-border bg-subtle px-3">
            <div className="flex items-center gap-2">
              <span className="grid size-7 place-items-center rounded-full bg-primary text-primary-foreground"><Bot className="size-4" /></span>
              <div><h2 className="text-xs font-bold uppercase tracking-wide">SkyGuard Assistant</h2><p className="text-[10px] text-muted-foreground">Operational network assistant</p></div>
            </div>
            <Button variant="ghost" size="icon" className="size-8" onClick={() => setOpen(false)} aria-label="Close chatbot"><X className="size-4" /></Button>
          </header>
          <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
            {messages.map((item, index) => (
              <div key={`${item.role}-${index}`} className={`flex ${item.role === "user" ? "justify-end" : "justify-start"}`}>
                <div className={item.role === "user" ? "max-w-[86%] rounded-md bg-primary px-3 py-2 text-xs text-primary-foreground" : "max-w-[86%] rounded-md border border-border bg-subtle px-3 py-2 text-xs"}>{item.text}</div>
              </div>
            ))}
            {sending && <div className="text-[11px] text-muted-foreground">SkyGuard is checking the network…</div>}
            <div ref={endRef} />
          </div>
          <form onSubmit={submit} className="flex gap-2 border-t border-border p-3">
            <Input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask about AWS_001…" disabled={sending} aria-label="Chat message" />
            <Button type="submit" size="icon" disabled={sending || !input.trim()} aria-label="Send message"><Send className="size-4" /></Button>
          </form>
        </section>
      )}
      <Button className="fixed bottom-4 right-4 z-[1200] shadow-panel" onClick={() => setOpen((value) => !value)} aria-label={open ? "Close SkyGuard Assistant" : "Open SkyGuard Assistant"}>
        <MessageCircle className="size-4" /><span className="hidden sm:inline">Assistant</span>
      </Button>
    </>
  );
}
