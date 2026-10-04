import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Lucid — EEG Analyzer",
  description: "Zero-cost public EEG research interface"
};

export default function RootLayout({
  children
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
