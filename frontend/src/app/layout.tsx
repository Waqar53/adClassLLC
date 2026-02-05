import type { Metadata } from 'next';
import '../styles/globals.css';

export const metadata: Metadata = {
    title: 'AdClass AI Platform',
    description: 'The AI System That Turns Ad Data Into Predictable Revenue Growth',
    keywords: 'advertising, AI, machine learning, ROAS optimization, creative prediction',
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en" className="dark">
            <body className="min-h-screen bg-neutral-950 antialiased">
                {children}
            </body>
        </html>
    );
}
