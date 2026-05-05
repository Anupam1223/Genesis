# Genesis Project - Interactive Documentation

This folder contains the interactive React-based documentation explorer for the Genesis Project. It is built using **Vite + React + Tailwind CSS (v4)**, making it lightning fast and easy to run locally on any machine.

## Prerequisites

Before running this documentation, ensure you have the following installed on your machine:
- **Node.js** (v18 or higher recommended)
- **npm** (comes with Node.js)

## Getting Started

If you have just cloned or downloaded this project onto a new device, follow these steps to get the documentation UI running:

1. **Navigate to the documentation directory:**
   ```bash
   cd documentation
   ```

2. **Install the dependencies:**
   ```bash
   npm install
   ```

3. **Start the local development server:**
   ```bash
   npm run dev
   ```

4. **View the site:**
   Open your browser and navigate to the local URL provided by Vite (usually `http://localhost:5173/`).

## Available Scripts

In the project directory, you can run:

- `npm run dev`: Starts the development server with Hot Module Replacement (HMR).
- `npm run build`: Compiles the application for production to the `dist` folder.
- `npm run preview`: Bootstraps a local web server that serves the production build from `dist`.

## Adding New Documentation Sections

To add a new slide or module to the interactive explorer:
1. Create a new component file inside `src/sections/` (e.g., `NewSection.jsx`).
2. Import the new component inside `src/App.jsx`.
3. Add a new entry to the `tabs` array in `App.jsx`, providing an `id`, a display `label`, and a Lucide `icon`.
4. Add the corresponding condition in the `<main>` rendering area:
   ```jsx
   {activeTab === 'new_id' && <NewSection />}
   ```