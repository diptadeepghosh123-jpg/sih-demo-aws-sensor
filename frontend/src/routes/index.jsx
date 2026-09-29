import { jsx as _jsx } from "react/jsx-runtime";
import { lazy, Suspense } from "react";
import { ClientOnly, createFileRoute } from "@tanstack/react-router";
const WeatherDashboard = lazy(() => import("@/components/weather-dashboard"));
export const Route = createFileRoute("/")({
    head: () => ({
        meta: [
            { title: "REDStone Miners | Anomaly Detection" },
            { name: "description", content: "Real-time GIS monitoring, anomaly alerts, and field sensor management for the Automatic Weather Station network." },
            { property: "og:title", content: "REDStone Miners" },
            { property: "og:description", content: "Real-time GIS monitoring and anomaly alerts for the Automatic Weather Station network." },
            { property: "og:type", content: "website" },
            { name: "twitter:card", content: "summary_large_image" },
        ],
    }),
    component: Index,
});
function Index() {
    return (_jsx(ClientOnly, { fallback: _jsx("div", { className: "flex h-dvh items-center justify-center bg-background text-sm font-medium text-muted-foreground", children: "Loading station network\u2026" }), children: _jsx(Suspense, { fallback: _jsx("div", { className: "flex h-dvh items-center justify-center bg-background text-sm font-medium text-muted-foreground", children: "Loading station network\u2026" }), children: _jsx(WeatherDashboard, {}) }) }));
}
